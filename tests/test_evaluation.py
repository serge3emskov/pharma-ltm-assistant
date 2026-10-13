"""Оценка по слоям: метрики и разбор ошибок по этапам на подстроенных ответах модели."""
import importlib.util
import json
from pathlib import Path

import pytest

from pharmaltm.devcheck import GoldLLM
from pharmaltm.evaluation import (
    ERR_CASCADE,
    ERR_CLASSIFIER,
    ERR_EXTRA,
    ERR_MISSED,
    ERR_NAMING,
    ERR_STATUS,
    ERR_SUBJECT,
    ERR_VALIDATION,
    GoldenSet,
    attribute_errors,
    check_answer,
    compare_reports,
    gold_backend,
    load_golden,
    run_golden,
    summarize,
    to_report,
    warm_cache,
)
from pharmaltm.llm import CachedLLM
from pharmaltm.prompts import compact_dialog, format_dialog

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bg", ROOT / "scripts" / "build_golden.py")
bg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bg)  # хелперы разметки: session, med, F, ...


def mini(steps_by_id: dict[str, list]) -> GoldenSet:
    return GoldenSet([{"id": k, "category": "multi_session" if len(v) > 1 else "test", "tags": ["t"],
                       "note": "", "steps": v} for k, v in steps_by_id.items()], [])


class Tampered(GoldLLM):
    """Эталонный бэкенд, у которого для заданных диалогов подменён ответ (task, section) → JSON."""

    def __init__(self, golden: GoldenSet, overrides: dict[tuple[str, str, str | None], dict]):
        base = gold_backend(golden)
        super().__init__([], [])
        self.dialogs, self.queries = base.dialogs, base.queries
        self.overrides = overrides

    def chat(self, messages, *, max_new_tokens=512, json_mode=False, meta=None):
        meta = meta or {}
        # диалог — в последнем сообщении пользователя с «Диалог:» (в повторе последним идёт инструкция)
        dialog = next((m["content"] for m in reversed(messages)
                       if m["role"] == "user" and m["content"].startswith("Диалог:")), messages[-1]["content"])
        for (text, task, section), answer in self.overrides.items():
            if text in dialog and meta.get("task") == task and section in (None, meta.get("section")):
                return json.dumps(answer, ensure_ascii=False)
        return super().chat(messages, max_new_tokens=max_new_tokens, json_mode=json_mode, meta=meta)


def errors_of(golden, overrides):
    run = run_golden(Tampered(golden, overrides), golden, run_answers=False)
    return run, {e["error"] for r in run.scenarios for e in attribute_errors(r)}


TXT = "Пью Детралекс по две таблетки."
SESSION = bg.session([bg.U(TXT)], [bg.med("Детралекс", "Пью Детралекс", status="active")],
                     expect=[bg.F("medications", "self", "детралекс")])


def test_classifier_miss_is_attributed_to_classifier():
    g = mini({"s": [SESSION]})  # «Детралекс» нет в справочнике — страховка не сработает
    run, errs = errors_of(g, {(TXT, "classify_sections", None): {"sections": []}})
    assert errs == {ERR_CLASSIFIER}
    s = summarize(run)
    assert s["classifier"]["recall"] == 0.0 and s["memory"]["facts"]["recall"] == 0.0


def test_extractor_miss():
    _, errs = errors_of(mini({"s": [SESSION]}), {(TXT, "extract", "medications"): {"facts": []}})
    assert errs == {ERR_MISSED}


def test_wrong_status_counted_once():
    run, errs = errors_of(mini({"s": [SESSION]}), {(TXT, "extract", "medications"): {"facts": [
        {"section": "medications", "name": "Детралекс", "status": "stopped", "evidence": "Пью Детралекс"}]}})
    assert errs == {ERR_STATUS}
    assert len([e for r in run.scenarios for e in attribute_errors(r)]) == 1  # пара missing/extra не удвоена


def test_wrong_subject_counted_once():
    run, errs = errors_of(mini({"s": [SESSION]}), {(TXT, "extract", "medications"): {"facts": [
        {"section": "medications", "subject": "мама", "name": "Детралекс", "evidence": "Пью Детралекс"}]}})
    assert errs == {ERR_SUBJECT} and len(attribute_errors(run.scenarios[0])) == 1


def test_naming_error():
    _, errs = errors_of(mini({"s": [SESSION]}), {(TXT, "extract", "medications"): {"facts": [
        {"section": "medications", "name": "Детралексом", "evidence": "Пью Детралекс"}]}})
    assert errs == {ERR_NAMING}


def test_validation_rejection():
    bad = {"facts": [{"section": "medications", "name": "Детралекс", "status": "принимает", "evidence": "Пью"}]}
    run, errs = errors_of(mini({"s": [SESSION]}), {(TXT, "extract", "medications"): bad,
                                                   (TXT, "extract_repair", "medications"): bad})
    assert errs == {ERR_VALIDATION}
    s = summarize(run)
    assert s["extractor"]["validation_repairs"] == 1 and s["extractor"]["rejected_validation"] == 1


def test_invented_evidence_is_rejected_by_code_before_memory():
    trap = bg.session([bg.U("Можно ли Терафлю при простуде?")], expect=[])
    run, errs = errors_of(mini({"t": [trap]}), {("Терафлю", "extract", "medications"): {"facts": [
        {"section": "medications", "name": "Терафлю", "evidence": "выдуманная цитата"}]}})
    assert errs == set()  # в память не попало
    s = summarize(run)
    assert s["extractor"]["events"]["precision"] == 0.0 and s["extractor"]["evidence_grounded"] == 0.0  # модель ошиблась
    assert s["extractor"]["rules"] == {"цитата не найдена в диалоге": 1}
    assert s["classifier"]["safety_net_added"] >= 1  # страховка открыла раздел по названию препарата


def test_extra_fact_from_question_reaches_memory():
    trap = bg.session([bg.U("Можно ли Терафлю при простуде?")], expect=[])
    run, errs = errors_of(mini({"t": [trap]}), {("Терафлю", "extract", "medications"): {"facts": [
        {"section": "medications", "name": "Терафлю", "evidence": "Можно ли Терафлю при простуде"}]}})
    assert errs == {ERR_EXTRA}


def test_missing_section_state():
    sess = bg.session([bg.A("Принимаете лекарства?"), bg.U("Нет.")],
                      statuses=[bg.st("medications", "confirmed_absent", "Нет.")],
                      expect=[], states=[bg.S("medications", "self", "confirmed_absent")])
    run, errs = errors_of(mini({"q": [sess]}), {("Принимаете", "extract", "medications"): {"facts": []}})
    assert errs == {ERR_MISSED}
    assert summarize(run)["memory"]["section_states"]["recall"] == 0.0


def test_cascade_from_previous_session():
    s1 = bg.session([bg.U("Пью Энап каждый день.")], [bg.med("Энап", "Пью Энап", status="active")],
                    expect=[bg.F("medications", "self", "эналаприл")])
    s2 = bg.session([bg.U("Энап больше не пью.")], [bg.med("Энап", "Энап больше не пью", pol="negative")],
                    expect=[bg.F("medications", "self", "эналаприл", "stopped")], after_days=3)
    run, errs = errors_of(mini({"c": [s1, s2]}), {("Пью Энап каждый день", "extract", "medications"): {"facts": []}})
    assert errs == {ERR_CASCADE}  # во второй сессии модель права; «denied» вместо «stopped» — из-за первой


def test_probe_checks_both_conditions():
    s1 = bg.session([bg.U("Пью Варфарин.")], [bg.med("Варфарин", "Пью Варфарин", status="active")],
                    expect=[bg.F("medications", "self", "варфарин")])
    p = bg.probe("Можно ли с моими таблетками Кеторол?", ["drug_compatibility"], include=[("medications", "варфарин")])
    g = mini({"p": [s1, p]})
    over = {("Можно ли с моими таблетками Кеторол?", "classify_query", None): {"query_types": ["general"]}}
    run = run_golden(Tampered(g, over), g, run_answers=False)
    s = summarize(run)
    assert s["retrieval"]["gold_query_types"]["probes_ok"] == 1  # логика выборки верна
    assert s["retrieval"]["llm"]["probes_ok"] == 0               # тип запроса LLM — нет
    assert s["retrieval"]["llm"]["query_types_exact"] == "0/1"


def test_answer_checks():
    checks = [{"kind": "mentions", "pattern": "аллерг", "why": ""},
              {"kind": "not_mentions", "pattern": r"\d+\s*мг", "why": ""}]
    assert [c["passed"] for c in check_answer("Учту АЛЛЕРГИЮ, доза — по инструкции.", checks)] == [True, True]
    assert [c["passed"] for c in check_answer("Пейте 400 мг.", checks)] == [False, False]


# ---------------------------------------------------------------- прогрев кэша и отчёты


class BatchGold(GoldLLM):
    def __init__(self, base):
        super().__init__([], [])
        self.dialogs, self.queries, self.name = base.dialogs, base.queries, "batch-gold"
        self.batches = []

    def chat_batch(self, messages_list, *, max_new_tokens=512, json_mode=False, metas=None, batch_size=8):
        self.batches.append(len(messages_list))
        return [self.chat(m, max_new_tokens=max_new_tokens, json_mode=json_mode, meta=meta)
                for m, meta in zip(messages_list, metas or [None] * len(messages_list))]


def test_warm_cache_makes_sequential_run_free(tmp_path):
    golden = load_golden()
    llm = CachedLLM(BatchGold(gold_backend(golden)), str(tmp_path / "cache.jsonl"))
    timings = warm_cache(llm, golden, batch_size=16, log=None)
    assert set(timings) == {"classify_sections", "classify_query", "extract", "extract_repair"}
    misses = llm.misses
    run = run_golden(llm, golden, run_answers=False)
    assert llm.misses == misses  # всё взято из кэша: ключи прогрева и пайплайна совпали
    assert max(llm.inner.batches) <= 16
    assert summarize(run)["memory"]["scenarios_ok"] == len(golden.scenarios)


def test_report_is_json_serializable_and_comparable():
    golden = load_golden()
    run = run_golden(gold_backend(golden), golden, run_answers=False)
    rep = json.loads(json.dumps(to_report(run, golden), ensure_ascii=False, default=list))
    assert rep["failures"] == [] and rep["golden"]["sessions"] >= 100
    rep2 = {**rep, "model": "other"}
    rows = compare_reports([rep, rep2])
    assert rows[0]["метрика"].startswith("Классификатор") and rows[0]["gold"] == rows[0]["other"] == 1.0


class BatchTampered(Tampered):
    def __init__(self, golden, overrides):
        super().__init__(golden, overrides)
        self.name = "batch-tampered"

    def chat_batch(self, messages_list, *, max_new_tokens=512, json_mode=False, metas=None, batch_size=8):
        return [self.chat(m, max_new_tokens=max_new_tokens, json_mode=json_mode, meta=meta)
                for m, meta in zip(messages_list, metas or [None] * len(messages_list))]


def test_warm_cache_batches_validation_repairs(tmp_path):
    g = mini({"s": [SESSION]})
    bad = {"facts": [{"section": "medications", "name": "Детралекс", "status": "принимает", "evidence": "Пью"}]}
    good = {"facts": [{"section": "medications", "name": "Детралекс", "status": "active", "evidence": "Пью"}]}
    llm = CachedLLM(BatchTampered(g, {(TXT, "extract", "medications"): bad,
                                      (TXT, "extract_repair", "medications"): good}), str(tmp_path / "c.jsonl"))
    logs = []
    warm_cache(llm, g, log=logs.append)
    assert any(line.startswith("extract_repair: 1 промптов") for line in logs)
    misses = llm.misses
    run = run_golden(llm, g, run_answers=False)
    assert llm.misses == misses and summarize(run)["extractor"]["validation_repaired"] == 1


def test_missed_fact_next_to_correct_other_subject_is_not_subject_error():
    """«Я не курю, а муж курит» → модель извлекла только мужа: это пропуск, а не ошибка субъекта."""
    txt = "Я не курю, а муж курит."
    sess = bg.session([bg.U(txt)], [bg.life("smoking", "нет", "Я не курю"), bg.life("smoking", "да", "а муж курит", subj="муж")],
                      expect=[bg.F("lifestyle", "self", "smoking"), bg.F("lifestyle", "муж", "smoking")])
    run, errs = errors_of(mini({"s": [sess]}), {(txt, "extract", "lifestyle"): {"facts": [
        {"section": "lifestyle", "subject": "муж", "attribute": "smoking", "value": "да", "evidence": "муж курит"}]}})
    assert errs == {ERR_MISSED}
