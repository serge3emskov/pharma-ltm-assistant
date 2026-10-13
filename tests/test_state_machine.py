"""Слой 3: детерминированные переходы состояния. LLM здесь не участвует."""
import pytest

from pharmaltm import ChangeType, POLICIES, Section, classify_transition
from conftest import USER, absent, ev, write

MED = POLICIES[Section.MEDICATIONS]


@pytest.mark.parametrize("old,new,changed,expected", [
    (None, "active", True, ChangeType.FIRST_MENTION),
    (None, "denied", False, ChangeType.DENIAL_RECORDED),
    ("active", "active", False, ChangeType.CONFIRMATION),
    ("active", "active", True, ChangeType.UPDATE),
    ("prescribed", "active", False, ChangeType.UPDATE),
    ("active", "stopped", False, ChangeType.DEACTIVATION),
    ("active", "denied", False, ChangeType.NEGATION),
    ("stopped", "active", False, ChangeType.REACTIVATION),
    ("stopped", "stopped", False, ChangeType.CONFIRMATION),
])
def test_transition_table(old, new, changed, expected):
    assert classify_transition(MED, old, new, changed) == expected


def test_medication_lifecycle_from_article(writer, store, clock):
    """Назначили → начал → сменили дозировку → перестал. История не теряется."""
    write(writer, ev("medications", "Энап", status="prescribed", indication="давление"))
    clock.advance(days=3)
    write(writer, ev("medications", "Энап 5 мг", status="active", frequency="утром"))
    clock.advance(days=20)
    write(writer, ev("medications", "Энап", dosage="10 мг"))
    clock.advance(days=10)
    write(writer, ev("medications", "Энап", polarity="negative"))

    fact = store.get_fact(USER, "self", Section.MEDICATIONS, "эналаприл")
    assert fact.status == "stopped" and fact.data["dosage"] == "10 мг" and fact.mentions == 4

    hist = store.history(fact.fact_id)
    assert [e.change_type for e in hist] == ["first_mention", "update", "update", "deactivation"]
    assert hist[1].after["data"]["dosage"] == "5 мг"       # дозировку из «Энап 5 мг» достал код
    assert hist[2].before["data"]["dosage"] == "5 мг"      # предыдущая дозировка — в истории
    assert hist[2].after["data"]["dosage"] == "10 мг"


def test_brand_and_inn_are_one_fact(writer, store):
    write(writer, ev("medications", "Энап"))
    r = write(writer, ev("medications", "эналаприл"))
    assert r.changes(ChangeType.CONFIRMATION)
    facts = store.list_facts(USER, "self", [Section.MEDICATIONS])
    assert len(facts) == 1 and facts[0].raw_names == ["Энап", "эналаприл"]


def test_confirmation_does_not_move_updated_at(writer, store, clock):
    write(writer, ev("medications", "Конкор"))
    created = store.get_fact(USER, "self", Section.MEDICATIONS, "бисопролол")
    clock.advance(days=5)
    write(writer, ev("medications", "Конкор"))
    f = store.get_fact(USER, "self", Section.MEDICATIONS, "бисопролол")
    assert f.updated_at == created.updated_at and f.last_confirmed_at > created.last_confirmed_at


def test_reactivation_after_stop(writer, store):
    write(writer, ev("medications", "Омез"))
    write(writer, ev("medications", "Омез", status="stopped"))
    r = write(writer, ev("medications", "омепразол"))
    assert r.applied[0].change_type == ChangeType.REACTIVATION
    assert store.get_fact(USER, "self", Section.MEDICATIONS, "омепразол").status == "active"


def test_negation_of_allergy(writer, store):
    write(writer, ev("allergies", "пенициллин", reaction="сыпь"))
    r = write(writer, ev("allergies", "пенициллин", polarity="negative"))
    assert r.applied[0].change_type == ChangeType.NEGATION
    f = store.get_fact(USER, "self", Section.ALLERGIES, "пенициллины")
    assert f.status == "denied" and f.data == {"reaction": "сыпь"}  # атрибуты не стираются, история цела


def test_denial_without_prior_fact(writer, store):
    r = write(writer, ev("special_states", "беременность", polarity="negative"))
    assert r.applied[0].change_type == ChangeType.DENIAL_RECORDED
    assert store.get_fact(USER, "self", Section.SPECIAL_STATES, "беременность").status == "denied"


def test_attribute_update_keeps_history(writer, store):
    write(writer, ev("demographics", attribute="weight_kg", value="24 кг", subject="дочке"))
    write(writer, ev("demographics", attribute="weight_kg", value=26, subject="дочь"))
    f = store.get_fact(USER, "дочь", Section.DEMOGRAPHICS, "weight_kg")
    assert f.data["value"] == 26.0
    assert [e.change_type for e in store.history(f.fact_id)] == ["first_mention", "update"]


# ---------------------------------------------------------------- «нет данных» vs «нет»


def test_unknown_vs_confirmed_absent_vs_dont_know(writer, store):
    assert store.get_section_state(USER, "self", Section.ALLERGIES).state is None          # не обсуждали
    write(writer, statuses=[absent("allergies")])
    assert store.get_section_state(USER, "self", Section.ALLERGIES).state == "confirmed_absent"
    write(writer, statuses=[absent("chronic_conditions", status="dont_know")])
    assert store.get_section_state(USER, "self", Section.CHRONIC_CONDITIONS).state == "dont_know"


def test_new_allergy_revokes_confirmed_absence(writer, store):
    write(writer, statuses=[absent("allergies")])
    r = write(writer, ev("allergies", "арахис"))
    assert r.changes(ChangeType.SECTION_STATE_CLEARED)
    assert store.get_section_state(USER, "self", Section.ALLERGIES).state is None


def test_denied_item_does_not_revoke_absence(writer, store):
    write(writer, statuses=[absent("allergies")])
    write(writer, ev("allergies", "пенициллин", polarity="negative"))
    assert store.get_section_state(USER, "self", Section.ALLERGIES).state == "confirmed_absent"


def test_absence_claim_with_active_allergy_is_conflict_not_deletion(writer, store):
    write(writer, ev("allergies", "пенициллин"))
    r = write(writer, statuses=[absent("allergies")])
    assert r.applied[0].change_type == ChangeType.CONFLICT
    st = store.get_section_state(USER, "self", Section.ALLERGIES)
    assert st.needs_clarification and st.state is None
    assert store.get_fact(USER, "self", Section.ALLERGIES, "пенициллины").status == "active"


def test_conflict_cleared_by_next_fact_in_section(writer, store):
    write(writer, ev("allergies", "пенициллин"))
    write(writer, statuses=[absent("allergies")])
    write(writer, ev("allergies", "пенициллин", evidence="да, на пенициллин сыпь была"))
    assert not store.get_section_state(USER, "self", Section.ALLERGIES).needs_clarification


def test_dont_know_does_not_override_known_facts(writer, store):
    write(writer, ev("chronic_conditions", "гипертония"))
    write(writer, statuses=[absent("chronic_conditions", status="dont_know")])
    assert store.get_section_state(USER, "self", Section.CHRONIC_CONDITIONS).state is None


def test_dont_know_downgrades_previous_absence(writer, store):
    write(writer, statuses=[absent("allergies")])
    write(writer, statuses=[absent("allergies", status="dont_know")])
    assert store.get_section_state(USER, "self", Section.ALLERGIES).state == "dont_know"


# ---------------------------------------------------------------- изоляция и валидация


def test_subjects_are_isolated(writer, store):
    write(writer, ev("allergies", "пенициллин", subject="дочке"))
    assert store.get_fact(USER, "дочь", Section.ALLERGIES, "пенициллины") is not None
    assert store.get_fact(USER, "self", Section.ALLERGIES, "пенициллины") is None


def test_invalid_event_not_written_and_logged(writer, store):
    r = write(writer, ev("medications", "Нурофен", status="taking"), ev("medications", "Омез"))
    assert len(r.rejected) == 1
    assert [f.key for f in store.list_facts(USER)] == ["омепразол"]
    assert store.rejections(USER)[0].kind == "fact"


def test_events_carry_payload_and_dialog(writer, store):
    write(writer, ev("medications", "Энап", evidence="пью Энап"), dialog="dialog-42")
    events = store.events(USER, "dialog-42")
    assert len(events) == 1 and events[0].payload["evidence"] == "пью Энап"


# ---------------------------------------------------------------- проверка по справочнику


def test_drug_in_diagnoses_is_rejected_by_reference(writer, store):
    r = write(writer, ev("reported_diagnoses", "Варфарин", subject="мама"), ev("reported_diagnoses", "цистит"))
    assert [f.key for f in store.list_facts(USER)] == ["цистит"]
    assert "справочнике препаратов" in r.rejected[0].errors[0]
    assert store.rejections(USER)[0].kind == "fact"


@pytest.mark.parametrize("section,name", [
    ("chronic_conditions", "давление"),      # есть в справочнике заболеваний
    ("allergies", "аспирин"),                # аллергия на препарат — нормально
    ("reported_diagnoses", "монурал-синдром"),  # нет ни в одном справочнике — решает не код
    ("symptoms", "кашель"),
])
def test_reference_check_keeps_legitimate_names(writer, store, section, name):
    r = write(writer, ev(section, name))
    assert not r.rejected and len(store.list_facts(USER)) == 1


# ---------------------------------------------------------------- защиты по ошибкам golden-прогона 1


def test_explicit_inactive_status_beats_negative_polarity(writer, store):
    write(writer, ev("symptoms", "изжога", polarity="negative", status="resolved"))
    assert store.get_fact(USER, "self", Section.SYMPTOMS, "изжога").status == "resolved"  # прошло, а не «не было»
    write(writer, ev("medications", "Энап", polarity="negative", status="stopped"))
    assert store.get_fact(USER, "self", Section.MEDICATIONS, "эналаприл").status == "stopped"


def test_plain_negation_still_denied(writer, store):
    write(writer, ev("symptoms", "кашель", polarity="negative"))
    assert store.get_fact(USER, "self", Section.SYMPTOMS, "кашель").status == "denied"


def test_negation_with_invalid_status_survives(writer, store):
    r = write(writer, ev("chronic_conditions", "гастрит", polarity="negative", status="confirmed_absent"))
    assert not r.rejected and store.get_fact(USER, "self", Section.CHRONIC_CONDITIONS, "гастрит").status == "denied"


def test_allergy_does_not_imply_not_taking(writer, store):
    r = write(writer, ev("allergies", "Амоксиклав", reaction="отёк"), ev("medications", "Амоксиклав", polarity="negative"))
    assert [f.section for f in store.list_facts(USER)] == [Section.ALLERGIES]
    assert "выведено из аллергии" in r.rejected[0].errors[0] and r.rejected[0].errors[0].startswith("код: ")


def test_med_negation_without_allergy_is_kept(writer, store):
    write(writer, ev("allergies", "пенициллин"), ev("medications", "Аспирин", polarity="negative"))
    assert store.get_fact(USER, "self", Section.MEDICATIONS, "ацетилсалициловая кислота").status == "denied"


@pytest.mark.parametrize("name", ["лекарства", "Лекарства.", "ни на что", "препараты"])
def test_generic_allergen_negation_becomes_section_status(writer, store, name):
    write(writer, ev("allergies", name, polarity="negative", evidence="Нет."))
    assert store.list_facts(USER) == []
    assert store.get_section_state(USER, "self", Section.ALLERGIES).state == "confirmed_absent"


@pytest.mark.parametrize("section,name,reason", [
    ("chronic_conditions", "аллергия на пенициллин", "аллергия"),
    ("symptoms", "аллергия", "аллергия"),
    ("symptoms", "простуда", "диагноз"),
])
def test_wrong_section_names_rejected(writer, store, section, name, reason):
    r = write(writer, ev(section, name))
    assert store.list_facts(USER) == [] and reason in r.rejected[0].errors[0]


def test_symptom_from_symptom_lexicon_not_treated_as_diagnosis(writer, store):
    write(writer, ev("symptoms", "температура"))
    assert len(store.list_facts(USER)) == 1


def test_evidence_must_be_found_in_dialog(writer, store):
    extraction = {"facts": [
        ev("special_states", "лактация", polarity="negative", evidence="не кормлю"),
        ev("medications", "Диротон", evidence="пьёт Диротон"),  # сокращённая, но настоящая цитата
    ]}
    r = writer.write(USER, extraction, "d", dialog_text="У папы давление, пьёт Сиофор и Диротон.")
    assert [f.key for f in store.list_facts(USER)] == ["лизиноприл"]
    assert "не найдена в диалоге" in r.rejected[0].errors[0]


def test_no_evidence_check_without_dialog(writer, store):
    write(writer, ev("medications", "Энап", evidence="что угодно"))
    assert len(store.list_facts(USER)) == 1


def test_stopped_med_derived_from_allergy_statement_is_dropped(writer, store):
    write(writer, ev("allergies", "Флемоксин", reaction="сыпь"), ev("medications", "Флемоксин", status="stopped"))
    write(writer, ev("allergies", "аспирин", polarity="negative"), ev("medications", "аспирин", polarity="negative"))
    assert {f.section for f in store.list_facts(USER)} == {Section.ALLERGIES}


def test_active_med_next_to_allergy_is_kept(writer, store):
    write(writer, ev("allergies", "пенициллин"), ev("medications", "Эриус", status="active"))
    assert store.get_fact(USER, "self", Section.MEDICATIONS, "дезлоратадин") is not None


def test_section_absence_naming_items_becomes_item_denials(writer, store):
    write(writer, statuses=[absent("chronic_conditions", evidence="Гастрита и язвы у меня нет")])
    got = {(f.key, f.status) for f in store.list_facts(USER)}
    assert got == {("гастрит", "denied"), ("язвенная болезнь желудка", "denied")}
    assert store.get_section_state(USER, "self", Section.CHRONIC_CONDITIONS).state is None  # про остальное не сказано


def test_plain_section_absence_unchanged(writer, store):
    write(writer, statuses=[absent("chronic_conditions", evidence="Хронических болезней нет")])
    assert store.get_section_state(USER, "self", Section.CHRONIC_CONDITIONS).state == "confirmed_absent"


def test_absence_item_also_stated_positive_is_not_denied(writer, store):
    write(writer, ev("chronic_conditions", "давление"),
          statuses=[absent("chronic_conditions", evidence="Хронических нет, только давление иногда")])
    assert store.get_fact(USER, "self", Section.CHRONIC_CONDITIONS, "артериальная гипертензия").status == "active"


@pytest.mark.parametrize("dialog,evidence,kept", [
    ("Лозап бросил месяц назад, сам.", "Лозап бросил месяц назад, сам.", False),
    ("Я не курю.", "Я не курю", True),
    ("Ассистент: Он курит? Покупатель: Да, много.", "Да, много.", True),
])
def test_lifestyle_needs_marker(writer, store, dialog, evidence, kept):
    writer.write(USER, {"facts": [ev("lifestyle", attribute="smoking", value="нет", evidence=evidence)]}, "d",
                 dialog_text=dialog)
    assert (len(store.list_facts(USER)) == 1) == kept


def test_allergy_reaction_is_not_current_symptom(writer, store):
    r = write(writer, ev("allergies", "Ибупрофен", reaction_type="intolerance", reaction="болит желудок"),
              ev("symptoms", "болит желудок"))
    assert [f.section for f in store.list_facts(USER)] == [Section.ALLERGIES]
    assert "реакция на препарат" in r.rejected[0].errors[0]
