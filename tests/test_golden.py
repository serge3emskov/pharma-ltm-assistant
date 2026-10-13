"""Golden set: разметка согласована с кодом, выборка отложена от настройки промптов, покрытие достаточное."""
import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest
from rapidfuzz import fuzz

from pharmaltm import QueryType, Section, parse_extraction
from pharmaltm.devcheck import load_dev
from pharmaltm.evaluation import gold_backend, load_golden, run_golden, summarize
from pharmaltm.lexicon import fold
from pharmaltm.prompts import CLASSIFIER_EXAMPLES, EXTRACTOR_EXAMPLES, QUERY_EXAMPLES

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = load_golden()
SESSIONS = list(GOLDEN.sessions())
PROBES = list(GOLDEN.probes())


@pytest.fixture(scope="module")
def gold_summary():
    return summarize(run_golden(gold_backend(GOLDEN), GOLDEN, run_answers=False))


def test_size():
    stats = GOLDEN.stats()
    assert stats["sessions"] >= 100 and stats["multi_session"] >= 10
    assert stats["probes"] >= 20 and stats["queries"] >= 40


@pytest.mark.parametrize("sid,step,session", SESSIONS, ids=lambda x: x if isinstance(x, str) else None)
def test_gold_extraction_valid_and_matches_sections(sid, step, session):
    facts, statuses, rejected = parse_extraction(session["gold_extraction"])
    assert not rejected, rejected
    assert {x.section for x in [*facts, *statuses]} == set(session["gold_sections"])


def test_gold_backend_is_100_percent_on_every_layer(gold_summary):
    s = gold_summary
    assert s["classifier"]["f1"] == 1.0
    assert s["extractor"]["events"]["f1"] == 1.0 and s["extractor"]["section_statuses"]["f1"] == 1.0
    assert s["extractor"]["field_accuracy"] == 1.0 and s["extractor"]["evidence_grounded"] == 1.0
    assert s["memory"]["facts"]["f1"] == 1.0 and s["memory"]["section_states"]["f1"] == 1.0
    assert s["memory"]["scenarios_ok"] == s["memory"]["scenarios"] and s["memory"]["data_errors"] == 0
    for cond in ("gold_query_types", "llm"):
        r = s["retrieval"][cond]
        assert r["probes_ok"] == r["probes"]
    assert s["queries"]["llm"]["types_exact_match"] == 1.0
    assert s["errors_by_stage"] == {}


def test_consistency_check_catches_bad_annotation():
    broken = copy.deepcopy(GOLDEN)
    sc = next(sc for sc in broken.scenarios if sc["id"] == "g-med-03")
    sc["steps"][0]["expect"]["facts"][0][3] = "active"  # на самом деле prescribed
    probe = next(st for sc in broken.scenarios for st in sc["steps"] if st["type"] == "probe")
    probe["context_include"].append(["medications", "несуществующий препарат"])
    s = summarize(run_golden(gold_backend(broken), broken, run_answers=False))
    assert s["memory"]["scenarios_ok"] == s["memory"]["scenarios"] - 1
    assert s["retrieval"]["gold_query_types"]["probes_ok"] == s["retrieval"]["gold_query_types"]["probes"] - 1


# ---------------------------------------------------------------- отложенность выборки


def _user_texts(dialogs):
    return [t["content"] for d in dialogs for t in d if t["role"] == "user" and len(t["content"]) > 12]


def test_no_near_duplicates_with_dev_set_and_prompt_examples():
    dev_dialogs, dev_queries = load_dev()
    tuning = _user_texts([d["turns"] for d in dev_dialogs])
    tuning += _user_texts([t for t, _ in CLASSIFIER_EXAMPLES])
    tuning += _user_texts([t for ex in EXTRACTOR_EXAMPLES.values() for t, _ in ex])
    golden = _user_texts([s["turns"] for _, _, s in SESSIONS])
    clashes = [(g, t) for g in golden for t in tuning if fuzz.ratio(fold(g), fold(t)) >= 80]
    assert not clashes

    tuning_q = [q["text"] for q in dev_queries] + [q for q, _ in QUERY_EXAMPLES]
    golden_q = [q["text"] for q in GOLDEN.queries] + [p["text"] for _, _, p in PROBES]
    clashes_q = [(g, t) for g in golden_q for t in tuning_q if fuzz.ratio(fold(g), fold(t)) >= 80]
    assert not clashes_q


# ---------------------------------------------------------------- покрытие и корректность разметки


def test_every_section_is_covered():
    counts = {s.value: 0 for s in Section}
    for _, _, s in SESSIONS:
        for sec in s["gold_sections"]:
            counts[sec] += 1
    assert min(counts.values()) >= 4, counts


def test_every_query_type_is_covered():
    counts = {q.value: 0 for q in QueryType}
    for q in GOLDEN.queries:
        for t in q["query_types"]:
            counts[t] += 1
    assert min(counts.values()) >= 2, counts


def test_probes_are_well_formed():
    sections = {s.value for s in Section}
    for _, _, p in PROBES:
        assert all(QueryType(t) for t in p["query_types"])
        for sec, _ in p["context_include"] + p["context_exclude"]:
            assert sec in sections
        for sec, reason, _ in p["clarify_include"]:
            assert sec in sections and reason in {"unknown", "conflict", "stale", "missing_attribute"}
        for c in p["answer_checks"]:
            re.compile(c["pattern"]) and c["kind"] in {"mentions", "not_mentions"}


def test_trap_scenarios_expect_no_invented_facts():
    traps = [sc for sc in GOLDEN.scenarios if any(t.startswith("trap_") for t in sc["tags"])]
    assert len(traps) >= 6
    assert any(sc["steps"][0]["expect"]["facts"] == [] for sc in traps)


def test_build_script_reproduces_data(tmp_path):
    spec = importlib.util.spec_from_file_location("build_golden", ROOT / "scripts" / "build_golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT = tmp_path
    mod.main()
    for name in ("scenarios.jsonl", "queries.jsonl"):
        assert (tmp_path / name).read_text(encoding="utf-8") == (ROOT / "data" / "golden" / name).read_text(
            encoding="utf-8")
