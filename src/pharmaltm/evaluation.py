"""Этап 3: оценка по слоям на golden set.

Как в статье, проверяется не только финальный ответ, а вся цепочка:

1. классификатор разделов — нашёл ли нужные разделы (LLM отдельно, страховка кодом отдельно);
2. extractor — события и их поля, отрицания, субъект, цитата evidence есть в диалоге;
3. память — итоговое состояние фактов после state machine (extractor + детерминированная логика);
4. retrieval — нужные факты попали в контекст, лишние не попали, правильные «уточнить»;
   считается дважды: с эталонным типом запроса (чистая логика выборки) и с LLM-классификатором;
5. E2E — факт сообщён в одной сессии, учтён в ответе в следующей (эвристические проверки ответа);
6. классификатор запроса — против baseline на правилах.

Каждая ошибка памяти приписывается этапу пайплайна: классификатор, пропуск extractor-а,
статус, субъект, нормализация названия, валидация, каскад из прошлой сессии.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from rapidfuzz import fuzz

from .devcheck import GoldLLM, QueryResult, _prf, run_query_dev, summarize_queries
from .lexicon import fold
from .llm import chat_many, extract_json_object
from .memory import CODE_RULE_PREFIX, MemoryWriter
from .normalize import Normalizer
from .pipeline import (
    VALIDATION_REPAIR_PROMPT,
    Extractor,
    LLMCall,
    QueryClassifier,
    SectionClassifier,
    WritePath,
    WriteResult,
    _validation_errors,
)
from .prompts import (
    answer_messages,
    compact_dialog,
    extractor_messages,
    query_classifier_messages,
    section_classifier_messages,
)
from .retrieval import ContextBuilder, ContextBundle
from .schemas import QueryClassification, QueryType, parse_extraction
from .sections import Section
from .store import MemoryStore

USER = "golden-user"
START = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
ENTITY_TEXT_FIELDS = ("dosage", "frequency", "indication", "reaction", "duration", "details")
ENUM_FIELDS = ("polarity", "status", "reaction_type", "severity", "diagnosed_by")
TEXT_MATCH = 70      # token_set_ratio для свободного текста в полях
GROUNDING_MATCH = 90  # partial_ratio: цитата evidence найдена в диалоге

# =========================================================================== данные


@dataclass
class GoldenSet:
    scenarios: list[dict]
    queries: list[dict]

    def sessions(self) -> Iterable[tuple[dict, int, dict]]:
        for sc in self.scenarios:
            for i, step in enumerate(sc["steps"]):
                if step["type"] == "session":
                    yield sc, i, step

    def probes(self) -> Iterable[tuple[dict, int, dict]]:
        for sc in self.scenarios:
            for i, step in enumerate(sc["steps"]):
                if step["type"] == "probe":
                    yield sc, i, step

    def stats(self) -> dict[str, Any]:
        cats = Counter(sc["category"] for sc in self.scenarios)
        tags = Counter(t for sc in self.scenarios for t in sc["tags"])
        return {
            "scenarios": len(self.scenarios),
            "multi_session": cats.get("multi_session", 0),
            "sessions": sum(1 for _ in self.sessions()),
            "probes": sum(1 for _ in self.probes()),
            "answer_checks": sum(len(p["answer_checks"]) for _, _, p in self.probes()),
            "queries": len(self.queries),
            "categories": dict(cats.most_common()),
            "tags": dict(tags.most_common()),
        }


def golden_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "golden"


def load_golden(directory: str | Path | None = None) -> GoldenSet:
    d = Path(directory) if directory else golden_dir()

    def read(name: str) -> list[dict]:
        with open(d / name, encoding="utf-8") as fh:
            return [json.loads(x) for x in fh if x.strip()]

    return GoldenSet(read("scenarios.jsonl"), read("queries.jsonl"))


def gold_backend(golden: GoldenSet) -> GoldLLM:
    """Эталонный бэкенд: отвечает разметкой golden set. С ним все слои должны дать 100%."""
    dialogs = [{"turns": s["turns"], "gold_sections": s["gold_sections"], "gold_extraction": s["gold_extraction"]}
               for _, _, s in golden.sessions()]
    queries = list(golden.queries) + [{"text": p["text"], "query_types": p["query_types"], "subject": p["subject"]}
                                      for _, _, p in golden.probes()]
    return GoldLLM(dialogs, queries)


# =========================================================================== прогрев кэша пакетами


def warm_cache(llm: Any, golden: GoldenSet, *, batch_size: int = 8, normalizer: Normalizer | None = None,
               log: Callable[[str], None] | None = print) -> dict[str, float]:
    """Заранее генерирует ответы пакетами, чтобы последовательный прогон взял их из кэша.

    Классификация и extraction не зависят от состояния памяти, поэтому их можно посчитать
    для всех сессий сразу: 1) классификаторы разделов и запросов; 2) extractor по разделам
    (с учётом страховки); 3) повторы extractor-а по ошибкам валидации. Сообщения строятся
    теми же функциями, что и в пайплайне, — ключи кэша совпадают дословно.
    """
    normalizer = normalizer or Normalizer()
    timings: dict[str, float] = {}

    def phase(name: str, messages: list, max_new_tokens: int, sections: list | None = None) -> list[str]:
        t0 = time.perf_counter()
        last = [0.0]

        def progress(done: int, total: int) -> None:
            if log and (time.perf_counter() - last[0] > 20 or done == total):
                last[0] = time.perf_counter()
                log(f"  {name}: {done}/{total}")

        # Те же метаданные, что передаёт пайплайн: их не видит модель, но по ним работают эталонный
        # и скриптованный бэкенды, а ключ кэша от метаданных не зависит
        metas = [{"task": name, "section": sec, "attempt": 1} for sec in (sections or [None] * len(messages))]
        outs = chat_many(llm, messages, max_new_tokens=max_new_tokens, json_mode=True,
                         metas=metas, batch_size=batch_size, on_progress=progress)
        timings[name] = round(time.perf_counter() - t0, 1)
        if log:
            log(f"{name}: {len(messages)} промптов, {timings[name]} с")
        return outs

    sessions = [compact_dialog(s["turns"]) for _, _, s in golden.sessions()]
    texts = [q["text"] for q in golden.queries] + [p["text"] for _, _, p in golden.probes()]
    classifier = SectionClassifier(llm)
    extractor = Extractor(llm)
    phase("classify_sections", [section_classifier_messages(t) for t in sessions], classifier.max_new_tokens)
    phase("classify_query", [query_classifier_messages(t) for t in texts], QueryClassifier(llm).max_new_tokens)

    write_path = WritePath(llm, MemoryWriter(MemoryStore(), normalizer))
    jobs: list[tuple[Section, list]] = []
    for turns in sessions:
        secs, call = classifier(turns)  # из кэша
        secs = list(Section) if not call.ok else secs + write_path.safety_net(turns, secs)
        jobs += [(s, extractor_messages(s, turns)) for s in secs]
    outs = phase("extract", [m for _, m in jobs], extractor.max_new_tokens, [s.value for s, _ in jobs])

    repairs, repair_sections = [], []
    for (section, msgs), raw in zip(jobs, outs):
        try:
            obj = extract_json_object(raw)
        except ValueError:
            continue  # повтор при битом JSON сделает последовательный прогон
        dummy = LLMCall("extract", section.value, raw, ok=True)
        facts, statuses, _ = Extractor._collect(obj, section, dummy)
        _, errors = _validation_errors(facts, statuses)
        if errors:
            repairs.append(msgs + [{"role": "assistant", "content": raw},
                                   {"role": "user", "content": VALIDATION_REPAIR_PROMPT.format(errors="\n".join(errors))}])
            repair_sections.append(section.value)
    phase("extract_repair", repairs, extractor.max_new_tokens, repair_sections)
    return timings


# =========================================================================== прогон


@dataclass
class SessionRun:
    scenario_id: str
    step: int
    gold: dict
    write: WriteResult
    facts: set[tuple]
    states: set[tuple]
    data_errors: list[str]
    gold_events: dict[tuple, dict]
    pred_events: dict[tuple, dict]
    gold_statuses: set[tuple]
    pred_statuses: set[tuple]
    rejected_keys: set[tuple]
    grounded: int
    pred_total: int


@dataclass
class ProbeRun:
    scenario_id: str
    step: int
    gold: dict
    classification: QueryClassification
    fallback: bool
    call_raw: str | None
    bundle_llm: ContextBundle
    bundle_gold: ContextBundle
    check_llm: dict
    check_gold: dict
    answer_prompt: list | None = None
    answer: str | None = None            # что увидит покупатель (после проверки по справочнику, если она есть)
    answer_results: list[dict] = field(default_factory=list)
    answer_raw: str | None = None        # ответ модели до проверки
    answer_results_raw: list[dict] = field(default_factory=list)
    post_check: Any = None               # PostCheck


@dataclass
class ScenarioRun:
    scenario: dict
    sessions: list[SessionRun] = field(default_factory=list)
    probes: list[ProbeRun] = field(default_factory=list)

    @property
    def final(self) -> SessionRun | None:
        with_expect = [s for s in self.sessions if s.gold.get("expect") is not None]
        return with_expect[-1] if with_expect else None

    def memory_ok(self) -> bool:
        f = self.final
        if f is None:
            return True
        exp = f.gold["expect"]
        return (f.facts == {tuple(x) for x in exp["facts"]} and f.states == {tuple(x) for x in exp["states"]}
                and not f.data_errors)


@dataclass
class EvalRun:
    model: str
    scenarios: list[ScenarioRun]
    queries_llm: list[QueryResult]
    queries_rules: list[QueryResult]
    timings: dict[str, float]
    cache: dict[str, int]


def _event_key(normalizer: Normalizer, ev: dict) -> tuple | None:
    section = ev.get("section")
    subj = normalizer.subject(ev.get("subject") or "self")
    if section in ("demographics", "lifestyle"):
        return (section, subj, ev.get("attribute"))
    name = ev.get("name")
    if not name or section not in {s.value for s in Section}:
        return None
    return (section, subj, normalizer.entity(Section(section), name).normalized)


def _events(normalizer: Normalizer, extraction: dict) -> tuple[dict, set, set]:
    """Валидные события (по ключу), статусы разделов и ключи отклонённых валидацией элементов."""
    facts, statuses, rejected = parse_extraction(extraction)
    events: dict[tuple, dict] = {}
    for ev in facts:
        d = ev.model_dump()
        key = _event_key(normalizer, d)
        if key and key not in events:
            events[key] = d
    sts = {(s.section, normalizer.subject(s.subject), s.status) for s in statuses}
    rej = set()
    for r in rejected:
        if isinstance(r.raw, dict):
            key = _event_key(normalizer, r.raw)
            if key:
                rej.add(key)
    return events, sts, rej


def _snapshot(store: MemoryStore) -> tuple[set, set]:
    facts = {(f.section.value, f.subject, f.key, f.status) for f in store.list_facts(USER)}
    states = {(s.section.value, s.subject, s.state) for s in store.list_section_states(USER) if s.state}
    return facts, states


def _norm_value(v: Any) -> str:
    return re.sub(r"\s+", "", fold(str(v)))


def _data_errors(store: MemoryStore, expect: dict) -> list[str]:
    errors = []
    for section, subj, key, fld, value in expect.get("data", []):
        fact = store.get_fact(USER, subj, Section(section), key)
        got = None if fact is None else fact.data.get(fld)
        if got is None or _norm_value(got) != _norm_value(value):
            errors.append(f"{section}/{subj}/{key}.{fld}: ожидалось {value!r}, получено {got!r}")
    return errors


def check_context(bundle: ContextBundle, probe: dict) -> dict:
    keys = {s.value: bundle.keys(s) for s in bundle.facts}
    clar = [(c.section.value, c.reason, c.key) for c in bundle.clarify]
    missing_include = [x for x in probe["context_include"] if x[1] not in keys.get(x[0], set())]
    present_exclude = [x for x in probe["context_exclude"] if x[1] in keys.get(x[0], set())]
    missing_clarify = [x for x in probe["clarify_include"]
                       if not any(c[0] == x[0] and c[1] == x[1] and (x[2] is None or c[2] == x[2]) for c in clar)]
    present_no_clarify = [s for s in probe["clarify_exclude"] if any(c[0] == s for c in clar)]
    return {
        "ok": not (missing_include or present_exclude or missing_clarify or present_no_clarify),
        "missing_include": missing_include, "present_exclude": present_exclude,
        "missing_clarify": missing_clarify, "present_no_clarify": present_no_clarify,
        "n_include": len(probe["context_include"]), "n_exclude": len(probe["context_exclude"]),
        "n_clarify": len(probe["clarify_include"]), "n_no_clarify": len(probe["clarify_exclude"]),
    }


def check_answer(answer: str, checks: list[dict]) -> list[dict]:
    text = fold(answer or "")
    out = []
    for c in checks:
        found = re.search(fold(c["pattern"]), text) is not None
        passed = found if c["kind"] == "mentions" else not found
        out.append({**c, "passed": passed})
    return out


def run_golden(
    llm: Any,
    golden: GoldenSet,
    *,
    normalizer: Normalizer | None = None,
    run_answers: bool = True,
    answer_max_new_tokens: int = 512,
    batch_size: int = 8,
    log: Callable[[str], None] | None = None,
    safety: Any = None,
) -> EvalRun:
    """safety — SafetyChecker: проверка по справочнику в контексте ответа и после ответа (этап 4).

    Без неё промпты ответов байт в байт совпадают с этапом 3, и ответы берутся из кэша.
    """
    normalizer = normalizer or Normalizer()
    t0 = time.perf_counter()
    scenario_runs: list[ScenarioRun] = []
    qclassifier = QueryClassifier(llm)
    for n, sc in enumerate(golden.scenarios, 1):
        clock_now = [START]
        clock = lambda: clock_now[0]  # noqa: E731
        store = MemoryStore()
        writer = MemoryWriter(store, normalizer, clock)
        builder = ContextBuilder(store, normalizer, clock, safety=safety)
        write_path = WritePath(llm, writer)
        run = ScenarioRun(sc)
        for i, step in enumerate(sc["steps"]):
            clock_now[0] += timedelta(days=step.get("after_days", 0))
            if step["type"] == "session":
                if step.get("setup"):
                    writer.write(USER, step["setup"], f"{sc['id']}#setup")
                write = write_path.process(USER, step["turns"], f"{sc['id']}#{i}")
                facts, states = _snapshot(store)
                gold_events, gold_statuses, _ = _events(normalizer, step["gold_extraction"])
                pred_events, pred_statuses, rejected_keys = _events(normalizer, write.extraction)
                dialog_text = fold("\n".join(t["content"] for t in write.turns))
                valid_f, valid_s, _ = parse_extraction(write.extraction)
                pred_all = [e.evidence for e in valid_f] + [s.evidence for s in valid_s]
                grounded = sum(1 for e in pred_all if fuzz.partial_ratio(fold(e), dialog_text) >= GROUNDING_MATCH)
                run.sessions.append(SessionRun(
                    sc["id"], i, step, write, facts, states,
                    _data_errors(store, step["expect"]) if step.get("expect") else [],
                    gold_events, pred_events, gold_statuses, pred_statuses, rejected_keys, grounded, len(pred_all),
                ))
            else:
                qc, call, fallback = qclassifier(step["text"])
                gold_qc = QueryClassification(query_types=[QueryType(t) for t in step["query_types"]],
                                              subject=step["subject"])
                b_llm = builder.build(USER, qc.query_types, qc.subject, reference_text=step["text"])
                b_gold = builder.build(USER, gold_qc.query_types, gold_qc.subject, reference_text=step["text"])
                pr = ProbeRun(sc["id"], i, step, qc, fallback, call.raw if call else None, b_llm, b_gold,
                              check_context(b_llm, step), check_context(b_gold, step))
                if run_answers and step["answer_checks"]:
                    pr.answer_prompt = answer_messages(b_llm.render(), [], step["text"])
                run.probes.append(pr)
        scenario_runs.append(run)
        if log and (n % 10 == 0 or n == len(golden.scenarios)):
            log(f"сценарии: {n}/{len(golden.scenarios)}")
    t_write = time.perf_counter() - t0

    t1 = time.perf_counter()
    pending = [p for r in scenario_runs for p in r.probes if p.answer_prompt]
    if pending:
        answers = chat_many(llm, [p.answer_prompt for p in pending], max_new_tokens=answer_max_new_tokens,
                            json_mode=False, metas=[{"task": "answer"}] * len(pending), batch_size=batch_size)
        for p, a in zip(pending, answers):
            p.answer_raw = p.answer = a.strip()
            if safety is not None and p.bundle_llm.safety is not None:
                p.post_check = safety.post_check(p.answer_raw, p.bundle_llm.safety.profile,
                                                 p.bundle_llm.safety.question_drugs)
                p.answer = p.post_check.text
            p.answer_results = check_answer(p.answer, p.gold["answer_checks"])
            p.answer_results_raw = check_answer(p.answer_raw, p.gold["answer_checks"])
    t_answers = time.perf_counter() - t1

    t2 = time.perf_counter()
    q_llm = run_query_dev(qclassifier, golden.queries, normalizer)
    q_rules = run_query_dev(None, golden.queries, normalizer)
    t_queries = time.perf_counter() - t2

    cache = {"hits": getattr(llm, "hits", 0), "misses": getattr(llm, "misses", 0)}
    return EvalRun(getattr(llm, "name", "unknown"), scenario_runs, q_llm, q_rules,
                   {"scenarios_s": round(t_write, 1), "answers_s": round(t_answers, 1),
                    "queries_s": round(t_queries, 1)}, cache)


# =========================================================================== метрики


def _rule_kind(error: str) -> str:
    text = error.removeprefix(CODE_RULE_PREFIX)
    for marker, kind in (("не найдена в диалоге", "цитата не найдена в диалоге"),
                         ("справочнике препаратов", "препарат не в своём разделе"),
                         ("аллергия, она относится", "аллергия не в своём разделе"),
                         ("а не симптом", "диагноз в симптомах"),
                         ("выведено из аллергии", "«не принимает» из аллергии"),
                         ("нет ничего про", "образ жизни без маркера в цитате"),
                         ("реакция на препарат из аллергии", "реакция из аллергии в симптомах")):
        if marker in text:
            return kind
    return "другое"


def _field_equal(fld: str, gold: Any, pred: Any) -> bool:
    if pred is None:
        return False
    if fld in ENTITY_TEXT_FIELDS:
        return fuzz.token_set_ratio(fold(str(gold)), fold(str(pred))) >= TEXT_MATCH
    if fld in ("pregnancy_week",):
        return int(gold) == int(pred)
    if fld == "value":
        try:
            return float(gold) == float(pred)
        except (TypeError, ValueError):
            return fold(str(gold)) == fold(str(pred))
    return fold(str(gold)) == fold(str(pred))


def _compare_fields(gold: dict, pred: dict) -> tuple[dict[str, tuple[int, int]], int]:
    """(верно, всего) по каждому полю эталона и число текстовых полей, которых в эталоне нет."""
    out: dict[str, tuple[int, int]] = {}
    entity = gold["section"] not in ("demographics", "lifestyle")
    fields = [f for f in gold if f not in ("section", "subject", "name", "evidence", "attribute")]
    if entity and "polarity" not in fields:
        fields.append("polarity")
    for fld in fields:
        g = gold.get(fld, "positive" if fld == "polarity" else None)
        if g is None:
            continue
        p = pred.get(fld, "positive" if fld == "polarity" and entity else None)
        out[fld] = (int(_field_equal(fld, g, p)), 1)
    extra = sum(1 for f in ENTITY_TEXT_FIELDS if pred.get(f) not in (None, "") and gold.get(f) is None)
    return out, extra


ERR_CLASSIFIER = "классификатор: раздел пропущен"
ERR_MISSED = "extractor: факт пропущен"
ERR_STATUS = "extractor: неверный статус или полярность"
ERR_SUBJECT = "extractor: неверный субъект"
ERR_NAMING = "extractor: название не нормализовано"
ERR_VALIDATION = "валидация: событие отклонено"
ERR_EXTRA = "extractor: лишний факт"
ERR_EXTRA_STATE = "extractor: лишний статус раздела"
ERR_CASCADE = "каскад: ошибка в предыдущей сессии"
ERR_LOGIC = "логика памяти"
ERR_OTHER = "прочее"


def _responsible(run: ScenarioRun, key3: tuple, upto: SessionRun) -> SessionRun | None:
    found = None
    for s in run.sessions:
        if key3 in s.gold_events:
            found = s
        if s is upto:
            break
    return found


def attribute_errors(run: ScenarioRun) -> list[dict]:
    final = run.final
    if final is None:
        return []
    sc = run.scenario
    exp_facts = {tuple(x) for x in final.gold["expect"]["facts"]}
    exp_states = {tuple(x) for x in final.gold["expect"]["states"]}
    missing, extra = exp_facts - final.facts, final.facts - exp_facts
    got3 = {f[:3]: f[3] for f in final.facts}
    exp3 = {f[:3]: f[3] for f in exp_facts}
    errors: list[dict] = []

    def err(kind, side, item, cat, session=None):
        errors.append({"scenario": sc["id"], "category_set": sc["category"], "tags": sc["tags"], "kind": kind,
                       "side": side, "item": list(item), "error": cat,
                       "session": session.step if session else None})

    naming_pairs: set[tuple] = set()
    for item in sorted(missing):
        key3 = item[:3]
        sess = _responsible(run, key3, final)
        if key3 in got3:
            if sess is None:
                err("fact", "missing", item, ERR_OTHER)
                continue
            pred = sess.pred_events.get(key3)
            gold = sess.gold_events[key3]
            same = pred is not None and all(
                fold(str(pred.get(f, "positive" if f == "polarity" else None)))
                == fold(str(gold.get(f, "positive" if f == "polarity" else None)))
                for f in ("polarity", "status") if f in gold or f == "polarity")
            if not same:
                err("fact", "missing", item, ERR_STATUS, sess)
            else:
                err("fact", "missing", item, ERR_CASCADE if len(run.sessions) > 1 else ERR_LOGIC, sess)
            continue
        if sess is None:
            err("fact", "missing", item, ERR_OTHER)
            continue
        section, subj, key = key3
        if Section(section) not in sess.write.sections:
            cat = ERR_CLASSIFIER
        elif any(k[0] == section and k[2] == key and k[1] != subj and k not in sess.gold_events
                 for k in sess.pred_events):
            cat = ERR_SUBJECT  # тот же факт приписан другому человеку (а не верно извлечён для другого)
        elif key3 in sess.rejected_keys:
            cat = ERR_VALIDATION
        elif any(k[0] == section and k[1] == subj and k not in sess.gold_events for k in sess.pred_events):
            cat = ERR_NAMING
            naming_pairs.add((section, subj))
        else:
            cat = ERR_MISSED
        err("fact", "missing", item, cat, sess)

    for item in sorted(extra):
        key3 = item[:3]
        if key3 in exp3:
            continue  # уже учтено как неверный статус
        section, subj, key = key3
        if (section, subj) in naming_pairs:
            continue  # пара к «название не нормализовано»
        if any(m[0] == section and m[2] == key and m[1] != subj for m in missing) and key3 not in exp3:
            continue  # пара к «неверный субъект»
        src = next((s for s in run.sessions if key3 in s.pred_events), None)
        err("fact", "extra", item, ERR_EXTRA, src)

    for item in sorted(exp_states - final.states):
        section, subj, _ = item
        src = None
        for s in run.sessions:
            if any(g[0] == section and g[1] == subj for g in s.gold_statuses):
                src = s
        if src is not None and Section(section) not in src.write.sections:
            cat = ERR_CLASSIFIER
        elif src is not None and any(p[0] == section and p[1] == subj for p in src.pred_statuses):
            cat = ERR_STATUS
        else:
            cat = ERR_MISSED
        err("state", "missing", item, cat, src)
    for item in sorted(final.states - exp_states):
        err("state", "extra", item, ERR_EXTRA_STATE)
    return errors


def summarize(run: EvalRun) -> dict[str, Any]:
    sessions = [s for r in run.scenarios for s in r.sessions]
    probes = [p for r in run.scenarios for p in r.probes]

    # ---- 1. классификатор разделов
    tp = fp = fn = 0
    per_section: dict[str, Counter] = defaultdict(Counter)
    routed_tp = routed_fn = 0
    net_added = net_recovered = 0
    for s in sessions:
        gold = set(s.gold["gold_sections"])
        pred = {x.value for x in s.write.classifier_sections}
        routed = {x.value for x in s.write.sections}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        routed_tp += len(gold & routed)
        routed_fn += len(gold - routed)
        for sec in gold | pred:
            per_section[sec]["tp"] += sec in gold and sec in pred
            per_section[sec]["fp"] += sec in pred and sec not in gold
            per_section[sec]["fn"] += sec in gold and sec not in pred
        net_added += len(s.write.safety_net)
        net_recovered += sum(1 for x in s.write.safety_net if x.value in gold)

    # ---- 2. extractor
    e_tp = e_fp = e_fn = 0
    st_tp = st_fp = st_fn = 0
    field_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    extra_attrs = 0
    grounded = pred_total = 0
    json_failures = repairs = repaired = rejected_validation = rejected_by_rules = off_section = 0
    rule_reasons: Counter = Counter()
    calls = 0
    for s in sessions:
        g, p = set(s.gold_events), set(s.pred_events)
        e_tp += len(g & p)
        e_fp += len(p - g)
        e_fn += len(g - p)
        for key in g & p:
            per_field, extra = _compare_fields(s.gold_events[key], s.pred_events[key])
            extra_attrs += extra
            for fld, (ok, total) in per_field.items():
                field_counts[fld][0] += ok
                field_counts[fld][1] += total
        st_tp += len(s.gold_statuses & s.pred_statuses)
        st_fp += len(s.pred_statuses - s.gold_statuses)
        st_fn += len(s.gold_statuses - s.pred_statuses)
        grounded += s.grounded
        pred_total += s.pred_total
        json_failures += sum(1 for c in s.write.calls if not c.ok)
        repairs += s.write.stats.repairs
        repaired += s.write.stats.repaired
        off_section += s.write.stats.off_section
        calls += len(s.write.calls)
        for r in s.write.report.rejected:
            if any(e.startswith(CODE_RULE_PREFIX) for e in r.errors):
                rejected_by_rules += 1
                rule_reasons[_rule_kind(r.errors[0])] += 1
            else:
                rejected_validation += 1
    fields_total = sum(v[1] for v in field_counts.values())
    fields_ok = sum(v[0] for v in field_counts.values())

    # ---- 3. память (итоговый снимок каждого сценария)
    m_tp = m_fp = m_fn = 0
    s_tp = s_fp = s_fn = 0
    data_errors = 0
    by_kind: dict[str, list[int]] = {"single": [0, 0], "multi_session": [0, 0]}
    for r in run.scenarios:
        f = r.final
        if f is None:
            continue
        exp_f = {tuple(x) for x in f.gold["expect"]["facts"]}
        exp_s = {tuple(x) for x in f.gold["expect"]["states"]}
        m_tp += len(exp_f & f.facts)
        m_fp += len(f.facts - exp_f)
        m_fn += len(exp_f - f.facts)
        s_tp += len(exp_s & f.states)
        s_fp += len(f.states - exp_s)
        s_fn += len(exp_s - f.states)
        data_errors += len(f.data_errors)
        kind = "multi_session" if r.scenario["category"] == "multi_session" else "single"
        by_kind[kind][0] += r.memory_ok()
        by_kind[kind][1] += 1

    # ---- 4. retrieval
    def retrieval(cond: str) -> dict[str, Any]:
        checks = [p.check_llm if cond == "llm" else p.check_gold for p in probes]
        n_inc = sum(c["n_include"] for c in checks)
        n_cl = sum(c["n_clarify"] for c in checks)
        return {
            "probes": len(checks),
            "probes_ok": sum(c["ok"] for c in checks),
            "include_recall": round(1 - sum(len(c["missing_include"]) for c in checks) / n_inc, 3) if n_inc else 1.0,
            "exclude_violations": sum(len(c["present_exclude"]) for c in checks),
            "clarify_recall": round(1 - sum(len(c["missing_clarify"]) for c in checks) / n_cl, 3) if n_cl else 1.0,
            "clarify_violations": sum(len(c["present_no_clarify"]) for c in checks),
        }

    probe_types_ok = sum({q.value for q in p.classification.query_types} == set(p.gold["query_types"])
                         for p in probes)

    # ---- 5. E2E-ответы
    answered = [p for p in probes if p.answer is not None]
    results = [c for p in answered for c in p.answer_results]
    results_raw = [c for p in answered for c in p.answer_results_raw]
    post = [p.post_check for p in answered if p.post_check is not None]

    # ---- ошибки по этапам
    errors = [e for r in run.scenarios for e in attribute_errors(r)]

    return {
        "model": run.model,
        "classifier": {
            **_prf(tp, fp, fn),
            "routed_recall": round(routed_tp / (routed_tp + routed_fn), 3) if routed_tp + routed_fn else 1.0,
            "safety_net_added": net_added,
            "safety_net_recovered": net_recovered,
            "per_section": {sec: {**_prf(c["tp"], c["fp"], c["fn"]), "support": c["tp"] + c["fn"]}
                            for sec, c in sorted(per_section.items())},
        },
        "extractor": {
            "events": _prf(e_tp, e_fp, e_fn),
            "section_statuses": _prf(st_tp, st_fp, st_fn),
            "field_accuracy": round(fields_ok / fields_total, 3) if fields_total else 1.0,
            "fields": {k: {"accuracy": round(v[0] / v[1], 3), "n": v[1]} for k, v in sorted(field_counts.items())},
            "extra_text_attributes": extra_attrs,
            "evidence_grounded": round(grounded / pred_total, 3) if pred_total else 1.0,
            "json_failures": json_failures,
            "validation_repairs": repairs,
            "validation_repaired": repaired,
            "rejected_validation": rejected_validation,
            "rejected_by_rules": rejected_by_rules,
            "rules": dict(rule_reasons.most_common()),
            "off_section_dropped": off_section,
        },
        "memory": {
            "facts": _prf(m_tp, m_fp, m_fn),
            "section_states": _prf(s_tp, s_fp, s_fn),
            "data_errors": data_errors,
            "scenarios_ok": sum(v[0] for v in by_kind.values()),
            "scenarios": sum(v[1] for v in by_kind.values()),
            "single_ok": f"{by_kind['single'][0]}/{by_kind['single'][1]}",
            "multi_session_ok": f"{by_kind['multi_session'][0]}/{by_kind['multi_session'][1]}",
        },
        "retrieval": {
            "gold_query_types": retrieval("gold"),
            "llm": {**retrieval("llm"), "query_types_exact": f"{probe_types_ok}/{len(probes)}"},
        },
        "e2e_answers": {
            "probes": len(answered),
            "checks": len(results),
            "passed": sum(c["passed"] for c in results),
            "rate": round(sum(c["passed"] for c in results) / len(results), 3) if results else None,
            "passed_raw": sum(c["passed"] for c in results_raw),
            "safety_checked": len(post),
            "answers_with_appended_warnings": sum(1 for x in post if x.appended),
            "appended_warnings": sum(len(x.appended) for x in post),
            "covered_warnings": sum(len(x.covered) + len(x.declined) for x in post),
            "answers_with_doses": sum(1 for x in post if x.doses),
        },
        "queries": {"llm": summarize_queries(run.queries_llm), "rules": summarize_queries(run.queries_rules)},
        "errors_by_stage": dict(Counter(e["error"] for e in errors).most_common()),
        "performance": {
            "write_llm_calls": calls,
            "calls_per_session": round(calls / len(sessions), 2) if sessions else 0,
            **run.timings,
            **{f"cache_{k}": v for k, v in run.cache.items()},
        },
    }


def tag_table(run: EvalRun) -> list[dict]:
    """Разрез по тегам и категориям: доля сценариев без ошибок памяти и полнота фактов."""
    rows: dict[str, Counter] = defaultdict(Counter)
    for r in run.scenarios:
        f = r.final
        if f is None:
            continue
        exp_f = {tuple(x) for x in f.gold["expect"]["facts"]}
        for label in [f"категория:{r.scenario['category']}"] + [f"тег:{t}" for t in r.scenario["tags"]]:
            c = rows[label]
            c["scenarios"] += 1
            c["ok"] += r.memory_ok()
            c["tp"] += len(exp_f & f.facts)
            c["fn"] += len(exp_f - f.facts)
            c["fp"] += len(f.facts - exp_f)
    out = []
    for label, c in sorted(rows.items(), key=lambda kv: (kv[1]["ok"] / kv[1]["scenarios"], kv[0])):
        out.append({"срез": label, "сценариев": c["scenarios"], "без ошибок": c["ok"],
                    "доля": round(c["ok"] / c["scenarios"], 2),
                    "полнота фактов": round(c["tp"] / (c["tp"] + c["fn"]), 2) if c["tp"] + c["fn"] else 1.0,
                    "лишних фактов": c["fp"]})
    return out


def query_type_table(results: list[QueryResult]) -> list[dict]:
    rows: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        for t in r.gold_types | r.pred_types:
            rows[t]["tp"] += t in r.gold_types and t in r.pred_types
            rows[t]["fp"] += t in r.pred_types and t not in r.gold_types
            rows[t]["fn"] += t in r.gold_types and t not in r.pred_types
    return [{"тип": t, **_prf(c["tp"], c["fp"], c["fn"]), "support": c["tp"] + c["fn"]} for t, c in sorted(rows.items())]


# =========================================================================== отчёт и сравнение


def to_report(run: EvalRun, golden: GoldenSet | None = None) -> dict[str, Any]:
    summary = summarize(run)
    failures = []
    for r in run.scenarios:
        if r.memory_ok():
            continue
        failures.append({
            "scenario": r.scenario["id"], "category": r.scenario["category"], "tags": r.scenario["tags"],
            "note": r.scenario.get("note", ""),
            "sessions": [{
                "step": s.step, "turns_seen": s.write.turns, "gold_sections": s.gold["gold_sections"],
                "classifier_sections": [x.value for x in s.write.classifier_sections],
                "safety_net": [x.value for x in s.write.safety_net],
                "gold_extraction": s.gold["gold_extraction"], "pred_extraction": s.write.extraction,
                "expected": s.gold.get("expect"), "facts": sorted(map(list, s.facts)),
                "states": sorted(map(list, s.states)), "data_errors": s.data_errors,
                "rejected": [{"raw": x.raw, "errors": x.errors} for x in s.write.report.rejected],
                "calls": [{"task": c.task, "section": c.section, "ok": c.ok, "attempts": c.attempts,
                           "error": c.error, "raw": c.raw} for c in s.write.calls],
            } for s in r.sessions],
        })
    probes = []
    for r in run.scenarios:
        for p in r.probes:
            probes.append({
                "scenario": p.scenario_id, "step": p.step, "text": p.gold["text"],
                "gold_types": p.gold["query_types"], "pred_types": [q.value for q in p.classification.query_types],
                "gold_subject": p.gold["subject"], "pred_subject": p.classification.subject,
                "fallback": p.fallback, "check_llm": p.check_llm, "check_gold": p.check_gold,
                "context_llm": p.bundle_llm.render() if not p.check_llm["ok"] or p.answer else None,
                "answer": p.answer, "answer_checks": p.answer_results,
                "answer_raw": p.answer_raw if p.answer_raw != p.answer else None,
                "answer_checks_raw": p.answer_results_raw if p.answer_raw != p.answer else None,
                "safety": p.bundle_llm.safety.as_dict() if p.bundle_llm.safety else None,
                "post_check": p.post_check.as_dict() if p.post_check else None,
            })
    return {
        "model": run.model,
        "created": datetime.now().isoformat(timespec="seconds"),
        "golden": golden.stats() if golden else None,
        "summary": summary,
        "tags": tag_table(run),
        "query_types": query_type_table(run.queries_llm),
        "errors": [e for r in run.scenarios for e in attribute_errors(r)],
        "failures": failures,
        "probes": probes,
        "queries": [{**q.row(), "raw": q.raw} for q in run.queries_llm],
    }


COMPARE_ROWS: list[tuple[str, Callable[[dict], Any]]] = [
    ("Классификатор разделов, F1", lambda s: s["classifier"]["f1"]),
    ("Классификатор + страховка, полнота", lambda s: s["classifier"]["routed_recall"]),
    ("Extractor: события, F1", lambda s: s["extractor"]["events"]["f1"]),
    ("Extractor: статусы разделов, F1", lambda s: s["extractor"]["section_statuses"]["f1"]),
    ("Extractor: точность полей", lambda s: s["extractor"]["field_accuracy"]),
    ("Extractor: evidence найдено в диалоге", lambda s: s["extractor"]["evidence_grounded"]),
    ("Память: факты, F1", lambda s: s["memory"]["facts"]["f1"]),
    ("Память: статусы разделов, F1", lambda s: s["memory"]["section_states"]["f1"]),
    ("Сценарии без ошибок памяти", lambda s: f"{s['memory']['scenarios_ok']}/{s['memory']['scenarios']}"),
    ("  одиночные", lambda s: s["memory"]["single_ok"]),
    ("  многосессионные", lambda s: s["memory"]["multi_session_ok"]),
    ("Retrieval (эталонный тип): пробы без ошибок",
     lambda s: f"{s['retrieval']['gold_query_types']['probes_ok']}/{s['retrieval']['gold_query_types']['probes']}"),
    ("Retrieval (E2E): пробы без ошибок",
     lambda s: f"{s['retrieval']['llm']['probes_ok']}/{s['retrieval']['llm']['probes']}"),
    ("Ответы: эвристики пройдены",
     lambda s: (f"{s['e2e_answers']['passed']}/{s['e2e_answers']['checks']}" if s["e2e_answers"]["checks"] else "—")),
    ("  до проверки по справочнику",
     lambda s: (f"{s['e2e_answers']['passed_raw']}/{s['e2e_answers']['checks']}"
                if s["e2e_answers"].get("safety_checked") else "—")),
    ("  ответов с дописанным предупреждением",
     lambda s: (f"{s['e2e_answers']['answers_with_appended_warnings']}/{s['e2e_answers']['safety_checked']}"
                if s["e2e_answers"].get("safety_checked") else "—")),
    ("Классификатор запроса: точное совпадение", lambda s: s["queries"]["llm"]["types_exact_match"]),
    ("  правила (baseline)", lambda s: s["queries"]["rules"]["types_exact_match"]),
    ("Вызовов LLM на сессию", lambda s: s["performance"]["calls_per_session"]),
]


def compare_reports(reports: list[dict]) -> list[dict]:
    rows = []
    for label, fn in COMPARE_ROWS:
        row = {"метрика": label}
        for rep in reports:
            try:
                row[rep["model"]] = fn(rep["summary"])
            except (KeyError, TypeError, ZeroDivisionError):
                row[rep["model"]] = None
        rows.append(row)
    return rows
