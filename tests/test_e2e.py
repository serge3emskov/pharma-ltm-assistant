"""Слой 5: E2E без LLM. Факт сообщён → извлечён (мок) → записан → найден в следующем диалоге."""
from pharmaltm import ChangeType, QueryType as Q, Section as S, classify_query_rules
from conftest import USER

DIALOG_1 = {  # «Пью Энап 5 мг по утрам от давления. Аллергий нет.»
    "facts": [
        {"section": "medications", "name": "Энап 5 мг", "status": "active", "frequency": "1 раз в день, утром",
         "indication": "повышенное давление", "evidence": "Пью Энап 5 мг по утрам от давления"},
        {"section": "chronic_conditions", "name": "давление", "evidence": "от давления"},
    ],
    "section_statuses": [{"section": "allergies", "status": "confirmed_absent", "evidence": "Аллергий нет"}],
}

DIALOG_2 = {  # «Энап больше не пью, врач перевёл на Лозап 50. И на пенициллин у меня сыпь.»
    "facts": [
        {"section": "medications", "name": "Энап", "polarity": "negative", "evidence": "Энап больше не пью"},
        {"section": "medications", "name": "Лозап 50 мг", "status": "active", "indication": "давление",
         "evidence": "врач перевёл на Лозап 50"},
        {"section": "allergies", "name": "пенициллин", "reaction": "сыпь", "evidence": "на пенициллин сыпь"},
    ]
}


def test_fact_survives_to_next_dialog(writer, builder, clock):
    writer.write(USER, DIALOG_1, "d1")
    clock.advance(days=7)
    q = classify_query_rules("Что выпить от головной боли?")
    bundle = builder.build(USER, q.query_types, q.subject)
    assert "эналаприл" in bundle.keys(S.MEDICATIONS)
    assert "артериальная гипертензия" in bundle.keys(S.CHRONIC_CONDITIONS)
    text = bundle.render()
    assert "эналаприл [принимает]" in text and "«Энап 5 мг»" in text and "подтверждено: нет" in text


def test_change_and_negation_flow_through(writer, builder, clock):
    writer.write(USER, DIALOG_1, "d1")
    clock.advance(days=30)
    report = writer.write(USER, DIALOG_2, "d2")
    assert report.counts() == {"deactivation": 1, "first_mention": 2, "section_state_cleared": 1}

    clock.advance(days=1)
    now = builder.build(USER, [Q.DRUG_COMPATIBILITY])
    assert now.keys(S.MEDICATIONS) == {"лозартан"}
    assert "пенициллины" in now.keys(S.ALLERGIES)

    history = builder.build(USER, [Q.THERAPY_HISTORY])
    assert history.keys(S.MEDICATIONS) == {"эналаприл", "лозартан"}


def test_rejected_output_does_not_touch_memory(writer, store):
    writer.write(USER, DIALOG_1, "d1")
    before = [(f.key, f.status) for f in store.list_facts(USER)]
    report = writer.write(USER, "Извините, я не могу ответить в формате JSON", "d2")
    assert report.rejected and not report.applied
    assert [(f.key, f.status) for f in store.list_facts(USER)] == before
    assert report.counts() == {"rejected": 1}


def test_report_rows_for_notebook(writer):
    rows = writer.write(USER, DIALOG_1, "d1").to_rows()
    assert {r["change"] for r in rows} == {ChangeType.FIRST_MENTION.value, ChangeType.SECTION_CONFIRMED_ABSENT.value}
