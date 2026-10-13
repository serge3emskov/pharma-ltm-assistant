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
