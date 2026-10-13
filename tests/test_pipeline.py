"""LLM-шаги на скриптованных ответах: всё, что вернула модель, проходит через код."""
import io
import json

import pytest

from pharmaltm import MemoryStore, MemoryWriter, QueryType, Section
from pharmaltm.llm import CachedLLM, OpenAICompatLLM, ScriptedLLM, extract_json_object
from pharmaltm.pipeline import (
    Extractor,
    PharmacyAssistant,
    QueryClassifier,
    SectionClassifier,
    WritePath,
    json_call,
)

TURNS = [{"role": "user", "content": "Пью Энап 5 мг. Аллергии на пенициллин нет."}]


def j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------- разбор JSON


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Вот ответ: {"a": 1} Надеюсь, помог.',
    '<think>рассуждаю {не json}</think>{"a": 1}',
    '{битый {"a": 1}',
])
def test_extract_json_object(text):
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_keeps_braces_inside_strings():
    assert extract_json_object('{"evidence": "скобка } внутри"}') == {"evidence": "скобка } внутри"}


@pytest.mark.parametrize("text", ["нет json", "[1, 2]", '{"a": '])
def test_extract_json_object_fails(text):
    with pytest.raises(ValueError):
        extract_json_object(text)


def test_json_call_retries_with_error_feedback():
    answers = iter(["Извините, не могу", '{"ok": true}'])
    llm = ScriptedLLM(lambda m, meta: next(answers))
    obj, call = json_call(llm, [{"role": "user", "content": "x"}], "t")
    assert obj == {"ok": True} and call.ok and call.attempts == 2
    retry_msgs = llm.calls[1]["messages"]
    assert retry_msgs[-2]["content"] == "Извините, не могу" and "JSON" in retry_msgs[-1]["content"]


def test_json_call_gives_up():
    llm = ScriptedLLM(lambda m, meta: "текст")
    obj, call = json_call(llm, [{"role": "user", "content": "x"}], "t")
    assert obj is None and not call.ok and call.attempts == 2


# ---------------------------------------------------------------- write-path


def test_section_classifier_filters_unknown_and_duplicates():
    llm = ScriptedLLM(lambda m, meta: j({"sections": ["allergies", "family_history", "allergies", "medications"]}))
    sections, call = SectionClassifier(llm)(TURNS)
    assert sections == [Section.ALLERGIES, Section.MEDICATIONS]
    assert call.ok and "family_history" in call.error


def test_extractor_drops_off_section_facts_and_fills_section():
    def responder(m, meta):
        return j({"facts": [
            {"name": "Энап 5 мг", "evidence": "Пью Энап 5 мг"},                                      # без section
            {"section": "allergies", "name": "пенициллин", "polarity": "negative", "evidence": "нет"},  # чужой
        ], "section_statuses": []})

    merged, calls, stats = Extractor(ScriptedLLM(responder))(TURNS, [Section.MEDICATIONS])
    assert [f["section"] for f in merged["facts"]] == ["medications"]
    assert stats.off_section == 1 and calls[0].parsed == 1


def test_extractor_calls_once_per_section_with_section_meta():
    llm = ScriptedLLM(lambda m, meta: j({"facts": [], "section_statuses": []}))
    Extractor(llm)(TURNS, [Section.MEDICATIONS, Section.ALLERGIES])
    assert [c["meta"]["section"] for c in llm.calls] == ["medications", "allergies"]


def _responder(sections, by_section):
    def respond(m, meta):
        if meta["task"] == "classify_sections":
            return sections
        return j(by_section.get(meta["section"], {"facts": [], "section_statuses": []}))
    return respond


def test_write_path_end_to_end():
    store = MemoryStore()
    llm = ScriptedLLM(_responder(j({"sections": ["medications", "allergies"]}), {
        "medications": {"facts": [{"section": "medications", "name": "Энап 5 мг", "status": "active",
                                   "evidence": "Пью Энап 5 мг"}]},
        "allergies": {"facts": [{"section": "allergies", "name": "пенициллин", "polarity": "negative",
                                 "evidence": "Аллергии на пенициллин нет"}]},
    }))
    res = WritePath(llm, MemoryWriter(store)).process("u", TURNS, "d1")
    assert {(f.key, f.status) for f in store.list_facts("u")} == {("эналаприл", "active"), ("пенициллины", "denied")}
    assert res.summary()["llm_calls"] == 3 and res.summary()["changes"] == {"first_mention": 1, "denial_recorded": 1}


def test_classifier_failure_falls_back_to_all_sections_and_logs():
    store = MemoryStore()
    llm = ScriptedLLM(_responder("не понимаю задачу", {}))
    res = WritePath(llm, MemoryWriter(store)).process("u", TURNS, "d1")
    assert res.classifier_failed and len(res.sections) == len(Section)
    assert sum(1 for c in llm.calls if c["meta"]["task"] == "extract") == len(Section)
    assert [r.kind for r in store.rejections("u")] == ["output"]


def test_broken_extractor_output_is_logged_not_written():
    store = MemoryStore()
    llm = ScriptedLLM(lambda m, meta: j({"sections": ["medications"]}) if meta["task"] == "classify_sections"
                      else "Энап — это эналаприл")
    res = WritePath(llm, MemoryWriter(store)).process("u", TURNS, "d1")
    assert store.list_facts("u") == [] and res.stats.failed_sections == 1
    assert store.rejections("u")[0].raw["section"] == "medications"


def test_invalid_event_from_llm_goes_through_validation():
    store = MemoryStore()
    llm = ScriptedLLM(_responder(j({"sections": ["medications"]}), {
        "medications": {"facts": [{"section": "medications", "name": "Энап", "status": "принимает",
                                   "evidence": "Пью Энап"}]},
    }))
    res = WritePath(llm, MemoryWriter(store)).process("u", TURNS, "d1")
    assert len(res.report.rejected) == 1 and store.list_facts("u") == []


# ---------------------------------------------------------------- read-path


def test_query_classifier_uses_llm():
    llm = ScriptedLLM(lambda m, meta: j({"query_types": ["dosing"], "subject": "дочка"}))
    qc, call, fallback = QueryClassifier(llm)("Сколько давать дочке?")
    assert qc.query_types == [QueryType.DOSING] and qc.subject == "дочка" and not fallback


def test_query_classifier_drops_unknown_types():
    llm = ScriptedLLM(lambda m, meta: j({"query_types": ["dosing", "diagnosis"], "subject": "self"}))
    qc, call, fallback = QueryClassifier(llm)("Как принимать?")
    assert qc.query_types == [QueryType.DOSING] and "diagnosis" in call.error and not fallback


@pytest.mark.parametrize("answer", ["не JSON", j({"query_types": ["diagnosis"]}), j({"subject": "self"})])
def test_query_classifier_falls_back_to_rules(answer):
    qc, call, fallback = QueryClassifier(ScriptedLLM(lambda m, meta: answer))("Что выпить от головной боли?")
    assert fallback and qc.query_types == [QueryType.SYMPTOM_RELIEF]


def test_query_classifier_without_llm_is_rules():
    qc, call, fallback = QueryClassifier(None)("Есть аналог подешевле?")
    assert fallback and call is None and qc.query_types == [QueryType.ANALOG]


# ---------------------------------------------------------------- ассистент и сессия


def test_session_remembers_between_dialogs():
    def respond(m, meta):
        task = meta["task"]
        if task == "classify_sections":
            return j({"sections": ["allergies"]})
        if task == "extract":
            return j({"facts": [{"section": "allergies", "name": "пенициллин", "evidence": "аллергия на пенициллин"}]})
        if task == "classify_query":
            return j({"query_types": ["symptom_relief"], "subject": "self"})
        return "ответ консультанта"

    llm = ScriptedLLM(respond)
    bot = PharmacyAssistant(llm)
    s1 = bot.session("u", "d1")
    assert s1.say("У меня аллергия на пенициллин") == "ответ консультанта"
    s1.close()

    s2 = bot.session("u", "d2")
    s2.say("Что выпить от горла?")
    system_prompt = [c for c in llm.calls if c["meta"]["task"] == "answer"][-1]["messages"][0]["content"]
    assert "пенициллины" in system_prompt and "ПАМЯТЬ АССИСТЕНТА" in system_prompt
    assert len(s2.turns) == 2 and s2.replies[0].read.classification.query_types == [QueryType.SYMPTOM_RELIEF]


# ---------------------------------------------------------------- бэкенды


def test_cached_llm(tmp_path):
    inner = ScriptedLLM(lambda m, meta: "ответ")
    path = tmp_path / "cache.jsonl"
    llm = CachedLLM(inner, str(path))
    msgs = [{"role": "user", "content": "x"}]
    llm.chat(msgs)
    llm.chat(msgs)
    assert len(inner.calls) == 1 and llm.hits == 1
    again = CachedLLM(inner, str(path))
    assert again.chat(msgs) == "ответ" and len(inner.calls) == 1
    assert again.chat(msgs, max_new_tokens=10) == "ответ" and len(inner.calls) == 2  # другой ключ


def test_openai_compat_request(monkeypatch):
    captured = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return Resp(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatLLM("http://localhost:11434/v1/", "qwen3:4b")
    assert llm.chat([{"role": "user", "content": "x"}], max_new_tokens=32, json_mode=True) == "{}"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["body"]["response_format"] == {"type": "json_object"} and captured["body"]["max_tokens"] == 32


# ---------------------------------------------------------------- повтор при невалидных событиях


def _extract_only(first, second=None):
    answers = iter([first] + ([second] if second else []))

    def respond(m, meta):
        if meta["task"] == "classify_sections":
            return j({"sections": ["medications"]})
        return next(answers)
    return respond


def test_validation_repair_is_accepted_when_better():
    store = MemoryStore()
    bad = j({"facts": [{"section": "medications", "name": "Энап", "status": "принимает", "evidence": "Пью Энап"}]})
    good = j({"facts": [{"section": "medications", "name": "Энап", "status": "active", "evidence": "Пью Энап"}]})
    llm = ScriptedLLM(_extract_only(bad, good))
    res = WritePath(llm, MemoryWriter(store)).process("u", TURNS, "d1")
    assert [f.key for f in store.list_facts("u")] == ["эналаприл"]
    assert res.stats.repairs == 1 and res.stats.repaired == 1
    repair_call = llm.calls[-1]
    assert repair_call["meta"]["task"] == "extract_repair" and "status" in repair_call["messages"][-1]["content"]


def test_validation_repair_rejected_when_it_loses_valid_events():
    store = MemoryStore()
    first = j({"facts": [
        {"section": "medications", "name": "Энап", "status": "active", "evidence": "Пью Энап"},
        {"section": "medications", "name": "Омез", "status": "принимает", "evidence": "и Омез"},
    ]})
    worse = j({"facts": []})
    llm = ScriptedLLM(_extract_only(first, worse))
    res = WritePath(llm, MemoryWriter(store)).process("u", TURNS, "d1")
    assert [f.key for f in store.list_facts("u")] == ["эналаприл"]  # исходный валидный факт сохранён
    assert res.stats.repairs == 1 and res.stats.repaired == 0 and len(res.report.rejected) == 1


def test_no_repair_when_everything_is_valid():
    good = j({"facts": [{"section": "medications", "name": "Энап", "evidence": "Пью Энап"}]})
    llm = ScriptedLLM(_extract_only(good))
    res = WritePath(llm, MemoryWriter(MemoryStore())).process("u", TURNS, "d1")
    assert res.stats.repairs == 0 and len(llm.calls) == 2


# ---------------------------------------------------------------- сжатие диалога и справочник в ответе


def test_write_path_hides_assistant_advice_from_extractor():
    turns = [
        {"role": "user", "content": "У меня гипертония, пью Лозап 50."},
        {"role": "assistant", "content": "Лозап обычно 5 мг, рекомендую Магний B6. Есть ли у вас аллергия?"},
        {"role": "user", "content": "Нет."},
    ]
    llm = ScriptedLLM(lambda m, meta: j({"sections": []}))
    res = WritePath(llm, MemoryWriter(MemoryStore())).process("u", turns, "d1")
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "Магний" not in prompt and "5 мг" not in prompt
    assert "Есть ли у вас аллергия?" in prompt and res.turns[1]["content"] == "Есть ли у вас аллергия?"


def test_answer_context_has_reference_for_mentioned_drugs():
    def respond(m, meta):
        if meta["task"] == "classify_query":
            return j({"query_types": ["general"], "subject": "self"})
        return "ок"

    llm = ScriptedLLM(respond)
    bot = PharmacyAssistant(llm)
    session = bot.session("u")
    session.say("Здравствуйте, пью Лозап 50 по утрам.")
    session.say("Спасибо.")
    system = [c for c in llm.calls if c["meta"]["task"] == "answer"][-1]["messages"][0]["content"]
    assert "СПРАВОЧНИК" in system and "лозартан" in system and "БРА" in system  # из истории, не из последней реплики


# ---------------------------------------------------------------- страховка классификатора разделов

LIVE_TURNS = [{"role": "user", "content": "У меня гипертония, пью Лозап 50 по утрам. На пенициллин аллергия."}]


def _router(sections):
    def respond(m, meta):
        if meta["task"] == "classify_sections":
            return j({"sections": sections})
        if meta["section"] == "medications":
            return j({"facts": [{"section": "medications", "name": "Лозап 50", "status": "active",
                                 "evidence": "пью Лозап 50 по утрам"}]})
        return j({"facts": [], "section_statuses": []})
    return respond


def test_safety_net_recovers_missed_medications():
    store = MemoryStore()
    llm = ScriptedLLM(_router(["chronic_conditions", "allergies"]))  # как в живом прогоне 2
    res = WritePath(llm, MemoryWriter(store)).process("u", LIVE_TURNS, "d1")
    assert res.classifier_sections == [Section.CHRONIC_CONDITIONS, Section.ALLERGIES]
    assert res.safety_net == [Section.MEDICATIONS] and res.summary()["safety_net"] == ["medications"]
    assert [f.key for f in store.list_facts("u")] == ["лозартан"]


def test_safety_net_silent_when_classifier_has_medications():
    res = WritePath(ScriptedLLM(_router(["medications"])), MemoryWriter(MemoryStore())).process("u", LIVE_TURNS)
    assert res.safety_net == [] and res.sections == [Section.MEDICATIONS]


def test_safety_net_silent_without_known_drug():
    turns = [{"role": "user", "content": "Пью витамин Д и рыбий жир."}]
    res = WritePath(ScriptedLLM(_router(["lifestyle"])), MemoryWriter(MemoryStore())).process("u", turns)
    assert res.safety_net == []


def test_safety_net_can_be_disabled():
    wp = WritePath(ScriptedLLM(_router(["allergies"])), MemoryWriter(MemoryStore()), safety_net=False)
    assert wp.process("u", LIVE_TURNS).safety_net == []


def test_safety_net_not_used_when_classifier_failed():
    llm = ScriptedLLM(lambda m, meta: "сбой" if meta["task"] == "classify_sections" else j({"facts": []}))
    res = WritePath(llm, MemoryWriter(MemoryStore())).process("u", LIVE_TURNS)
    assert res.classifier_failed and res.safety_net == [] and len(res.sections) == len(Section)


def test_answer_prompt_forbids_doses():
    from pharmaltm.prompts import answer_messages
    system = answer_messages("ctx", [], "Сколько пить парацетамола?")[0]["content"]
    assert "Не называй дозы" in system


@pytest.mark.parametrize("text,expected", [
    ("Сахарный диабет 2 типа, на диете, таблеток не пью.", ["medications"]),
    ("Беременна, лекарства никакие не пью.", ["medications"]),
    ("Ничего другого не пью.", ["medications"]),
    ("Никаких лекарств постоянно не принимаю.", ["medications"]),
    ("Думаю, у меня обычная простуда.", ["reported_diagnoses"]),
    ("Алкоголь не пью вообще.", []),
    ("Болит горло.", []),
])
def test_safety_net_patterns(text, expected):
    wp = WritePath(ScriptedLLM(lambda m, meta: "{}"), MemoryWriter(MemoryStore()))
    assert [s.value for s in wp.safety_net([{"role": "user", "content": text}], [])] == expected


def test_write_path_checks_evidence_against_dialog():
    turns = [{"role": "user", "content": "Беременна, 20 недель."}]

    def respond(m, meta):
        if meta["task"] == "classify_sections":
            return j({"sections": ["special_states"]})
        return j({"facts": [
            {"section": "special_states", "name": "беременность", "pregnancy_week": 20, "evidence": "Беременна, 20 недель"},
            {"section": "special_states", "name": "лактация", "polarity": "negative", "evidence": "не кормлю"}]})

    store = MemoryStore()
    res = WritePath(ScriptedLLM(respond), MemoryWriter(store)).process("u", turns, "d")
    assert [f.key for f in store.list_facts("u")] == ["беременность"] and len(res.report.rejected) == 1
