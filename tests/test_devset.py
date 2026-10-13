"""Dev-набор согласован с кодом: эталонные ответы дают ровно ожидаемое состояние памяти."""
import pytest

from pharmaltm import parse_extraction
from pharmaltm.devcheck import (
    GoldLLM,
    load_dev,
    run_query_dev,
    run_write_dev,
    summarize_queries,
    summarize_write,
)
from pharmaltm.pipeline import QueryClassifier

DIALOGS, QUERIES = load_dev()


@pytest.mark.parametrize("dialog", DIALOGS, ids=lambda d: d["id"])
def test_gold_extraction_is_valid_and_matches_gold_sections(dialog):
    facts, statuses, rejected = parse_extraction(dialog["gold_extraction"])
    assert not rejected
    assert {x.section for x in [*facts, *statuses]} == set(dialog["gold_sections"])


def test_gold_backend_reproduces_expected_memory():
    results = run_write_dev(GoldLLM(DIALOGS, QUERIES), DIALOGS)
    failed = [r.row() for r in results if not r.ok]
    assert not failed
    summary = summarize_write(results, DIALOGS)
    assert summary["facts"]["f1"] == 1.0 and summary["section_states"]["f1"] == 1.0


def test_gold_backend_reproduces_query_labels():
    summary = summarize_queries(run_query_dev(QueryClassifier(GoldLLM(DIALOGS, QUERIES)), QUERIES))
    assert summary["types_exact_match"] == 1.0 and summary["subject_accuracy"] == 1.0


def test_safety_net_fires_only_where_drug_is_mentioned_outside_gold_sections():
    results = run_write_dev(GoldLLM(DIALOGS, QUERIES), DIALOGS)
    fired = {r.id for r in results if r.write.safety_net}
    assert fired == {"d02", "d14"}  # препарат только в вопросе — extractor должен вернуть пусто


def test_rules_baseline_is_measurable():
    summary = summarize_queries(run_query_dev(None, QUERIES))
    assert 0.5 < summary["types_exact_match"] < 1.0  # dev-набор не решается одними правилами


def test_dev_set_covers_every_section():
    covered = {s for d in DIALOGS for s in d["gold_sections"]}
    from pharmaltm import Section
    assert covered == {s.value for s in Section}
