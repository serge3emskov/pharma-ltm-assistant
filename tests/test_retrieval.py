"""Слой 4: в контекст попадают нужные разделы и статусы — и не попадает лишнее."""
from pharmaltm import PurchaseRecord, QueryType as Q, Section as S, select_sections, statuses_for
from conftest import absent, ev, write


def test_sections_union_without_duplicates():
    secs = select_sections([Q.ANALOG, Q.DRUG_COMPATIBILITY])
    assert secs[:2] == [S.MEDICATIONS, S.ALLERGIES] and len(secs) == len(set(secs))


def test_general_query_has_no_personal_context(writer, builder):
    write(writer, ev("medications", "Энап"))
    b = builder.build("u-test", [Q.GENERAL])
    assert b.sections == [] and not b.keys()
    assert "не нужен" in b.render()


def test_irrelevant_sections_not_in_context(writer, builder):
    write(writer, ev("medications", "Энап"), ev("symptoms", "кашель"),
          ev("lifestyle", attribute="driving", value="да"))
    b = builder.build("u-test", [Q.ANALOG])
    assert b.sections_with_facts() == {S.MEDICATIONS}
    assert "кашель" not in b.render()


def test_stopped_drug_excluded_for_current_question_but_kept_for_history(writer, builder):
    write(writer, ev("medications", "Энап", status="stopped"), ev("medications", "Лозап"))
    assert builder.build("u-test", [Q.DRUG_COMPATIBILITY]).keys(S.MEDICATIONS) == {"лозартан"}
    assert builder.build("u-test", [Q.THERAPY_HISTORY]).keys(S.MEDICATIONS) == {"эналаприл", "лозартан"}


def test_statuses_rules():
    assert "stopped" not in statuses_for(S.MEDICATIONS, [Q.SYMPTOM_RELIEF])
    assert "stopped" in statuses_for(S.MEDICATIONS, [Q.SYMPTOM_RELIEF, Q.THERAPY_HISTORY])
    assert "denied" in statuses_for(S.ALLERGIES, [Q.ANALOG])          # «аллергии на X нет» — полезно
    assert "denied" not in statuses_for(S.MEDICATIONS, [Q.THERAPY_HISTORY])


def test_denied_allergy_rendered_as_confirmed_absence(writer, builder):
    write(writer, ev("allergies", "пенициллин", polarity="negative"))
    text = builder.build("u-test", [Q.ANALOG]).render()
    assert "пенициллины: НЕТ (подтверждено отсутствие)" in text


# ---------------------------------------------------------------- забывание по типу факта


def test_old_symptom_is_forgotten_for_context(writer, builder, clock):
    write(writer, ev("symptoms", "кашель"))
    clock.advance(days=15)
    assert "кашель" not in builder.build("u-test", [Q.SYMPTOM_RELIEF]).keys()
    assert "кашель" in builder.build("u-test", [Q.PROFILE_REVIEW]).keys()   # история не удалена


def test_old_allergy_never_expires(writer, builder, clock):
    write(writer, ev("allergies", "пенициллин"))
    clock.advance(days=3 * 365)
    assert "пенициллины" in builder.build("u-test", [Q.SYMPTOM_RELIEF]).keys()


def test_old_pregnancy_is_flagged_not_dropped(writer, builder, clock):
    write(writer, ev("special_states", "беременность", pregnancy_week=12))
    clock.advance(days=300)
    b = builder.build("u-test", [Q.SYMPTOM_RELIEF])
    assert "беременность" in b.keys(S.SPECIAL_STATES)
    assert any(c.reason == "stale" and c.key == "беременность" for c in b.clarify)
    assert "давно не подтверждалось" in b.render()


def test_reconfirmation_resets_staleness(writer, builder, clock):
    write(writer, ev("symptoms", "кашель"))
    clock.advance(days=10)
    write(writer, ev("symptoms", "кашель"))
    clock.advance(days=10)
    assert "кашель" in builder.build("u-test", [Q.SYMPTOM_RELIEF]).keys()


# ---------------------------------------------------------------- что уточнить у покупателя


def _clarify(bundle):
    return {(c.section, c.reason, c.key) for c in bundle.clarify}


def test_unknown_critical_sections_requested(builder):
    got = _clarify(builder.build("u-test", [Q.SYMPTOM_RELIEF]))
    assert {(S.ALLERGIES, "unknown", None), (S.SPECIAL_STATES, "unknown", None)} <= got


def test_confirmed_absence_and_dont_know_not_requested_again(writer, builder):
    write(writer, statuses=[absent("allergies"), absent("medications", status="dont_know")])
    got = _clarify(builder.build("u-test", [Q.SYMPTOM_RELIEF]))
    assert not {s for s, _, _ in got} & {S.ALLERGIES, S.MEDICATIONS}


def test_only_denied_item_still_leaves_allergies_unknown(writer, builder):
    write(writer, ev("allergies", "пенициллин", polarity="negative"))
    assert (S.ALLERGIES, "unknown", None) in _clarify(builder.build("u-test", [Q.SYMPTOM_RELIEF]))


def test_not_pregnant_answers_special_states(writer, builder):
    write(writer, ev("special_states", "беременность", polarity="negative"))
    assert S.SPECIAL_STATES not in {s for s, _, _ in _clarify(builder.build("u-test", [Q.SYMPTOM_RELIEF]))}


def test_pregnancy_not_asked_for_men_and_children(writer, builder):
    write(writer, ev("demographics", attribute="sex", value="м"))
    assert S.SPECIAL_STATES not in {s for s, _, _ in _clarify(builder.build("u-test", [Q.SYMPTOM_RELIEF]))}
    kid = builder.build("u-test", [Q.SYMPTOM_RELIEF], subject="ребёнку")
    assert S.SPECIAL_STATES not in {s for s, _, _ in _clarify(kid)}


def test_dosing_for_child_requires_age_and_weight(builder):
    got = _clarify(builder.build("u-test", [Q.DOSING], subject="ребёнок"))
    assert {(S.DEMOGRAPHICS, "missing_attribute", "age"),
            (S.DEMOGRAPHICS, "missing_attribute", "weight_kg")} <= got


def test_conflict_is_surfaced(writer, builder):
    write(writer, ev("allergies", "пенициллин"))
    write(writer, statuses=[absent("allergies")])
    b = builder.build("u-test", [Q.ANALOG])
    assert (S.ALLERGIES, "conflict", None) in _clarify(b)
    assert "пенициллины" in b.keys()  # спорный факт не скрывается


def test_profile_review_has_no_clarifications(builder):
    assert builder.build("u-test", [Q.PROFILE_REVIEW]).clarify == []


# ---------------------------------------------------------------- источники с пометкой происхождения


def test_purchase_history_is_a_separate_labelled_block(writer, builder):
    write(writer, ev("medications", "Энап", polarity="negative", evidence="от давления ничего не пью"))
    purchases = [PurchaseRecord("2026-08-20", "Диротон 10 мг")]
    text = builder.build("u-test", [Q.DRUG_COMPATIBILITY], purchases=purchases).render()
    mem = text.index("со слов покупателя")
    doc = text.index("ДОКУМЕНТИРОВАНО")
    assert mem < doc and "→ лизиноприл" in text
    assert "уточни" in text  # расхождение не разрешается кодом, решение — за диалогом


def test_purchases_skipped_when_medications_not_needed(builder):
    b = builder.build("u-test", [Q.GENERAL], purchases=[PurchaseRecord("2026-08-20", "Омез")])
    assert b.purchases == []


# ---------------------------------------------------------------- справочник в контексте


def test_reference_block_for_mentioned_drugs(builder):
    b = builder.build("u-test", [Q.DRUG_COMPATIBILITY], reference_text="Можно ли ТераФлю с Конкором?")
    text = b.render()
    assert "### СПРАВОЧНИК" in text
    assert "«терафлю» → комбинированный препарат: парацетамол, фенилэфрин, фенирамин" in text
    assert "«конкором» → бисопролол; группа: бета-адреноблокаторы" in text


def test_reference_block_even_without_personal_context(builder):
    text = builder.build("u-test", [Q.GENERAL], reference_text="Есть Смекта?").render()
    assert "не нужен" in text and "диосмектит" in text


def test_clarify_header_tells_not_to_reask(builder):
    text = builder.build("u-test", [Q.SYMPTOM_RELIEF]).render()
    assert "не переспрашивай" in text
