"""Слой 1: контракт extractor-а. Невалидное событие не проходит, валидные соседи — проходят."""
import json

import pytest
from pydantic import ValidationError

from pharmaltm.schemas import (
    DemographicsEvent,
    QueryClassification,
    SpecialStateEvent,
    parse_extraction,
)
from conftest import absent, ev


def test_valid_medication_event_passes():
    facts, statuses, rej = parse_extraction({"facts": [ev("medications", "Энап 5 мг", status="active")]})
    assert len(facts) == 1 and not rej
    assert facts[0].name == "Энап 5 мг" and facts[0].polarity == "positive" and facts[0].subject == "self"


def test_extra_field_rejected():
    _, _, rej = parse_extraction({"facts": [ev("medications", "Энап", fact_id="123")]})
    assert len(rej) == 1 and "fact_id" in rej[0].errors[0]


def test_invalid_status_rejected_but_neighbours_kept():
    facts, _, rej = parse_extraction({"facts": [
        ev("medications", "Нурофен", status="taking"),
        ev("allergies", "пенициллин"),
    ]})
    assert [f.section for f in facts] == ["allergies"]
    assert rej[0].kind == "fact" and "status" in rej[0].errors[0]


def test_unknown_section_rejected():
    _, _, rej = parse_extraction({"facts": [ev("family_history", "диабет у мамы")]})
    assert len(rej) == 1


def test_missing_evidence_rejected():
    _, _, rej = parse_extraction({"facts": [{"section": "medications", "name": "Энап"}]})
    assert len(rej) == 1 and "evidence" in rej[0].errors[0]


def test_section_status_only_for_absence_aware_sections():
    _, statuses, rej = parse_extraction({"section_statuses": [
        absent("allergies"),
        absent("symptoms"),  # «симптомов нет» на уровне раздела не храним
    ]})
    assert [s.section for s in statuses] == ["allergies"] and len(rej) == 1


def test_json_string_with_markdown_fence():
    raw = "```json\n" + json.dumps({"facts": [ev("allergies", "арахис")]}, ensure_ascii=False) + "\n```"
    facts, _, rej = parse_extraction(raw)
    assert len(facts) == 1 and not rej


def test_broken_json_is_one_output_rejection():
    facts, statuses, rej = parse_extraction('{"facts": [')
    assert not facts and not statuses and rej[0].kind == "output"


def test_unknown_top_level_key_reported():
    _, _, rej = parse_extraction({"facts": [], "memory_update": "drop all"})
    assert rej and rej[0].kind == "output"


@pytest.mark.parametrize("value,expected", [("45 лет", 45), (7, 7), ("7.5", 7)])
def test_age_parsed(value, expected):
    e = DemographicsEvent(section="demographics", attribute="age", value=value, evidence="x")
    assert e.value == expected


@pytest.mark.parametrize("value", [150, "много", -1])
def test_age_out_of_range_rejected(value):
    with pytest.raises(ValidationError):
        DemographicsEvent(section="demographics", attribute="age", value=value, evidence="x")


def test_sex_normalized():
    e = DemographicsEvent(section="demographics", attribute="sex", value="Женский", evidence="x")
    assert e.value == "ж"


def test_special_state_synonyms_and_week_rule():
    assert SpecialStateEvent(section="special_states", name="кормлю грудью", evidence="x").name == "лактация"
    with pytest.raises(ValidationError):
        SpecialStateEvent(section="special_states", name="лактация", pregnancy_week=10, evidence="x")
    with pytest.raises(ValidationError):
        SpecialStateEvent(section="special_states", name="беременность", pregnancy_week=60, evidence="x")


def test_null_and_empty_fields_mean_absent():
    facts, _, rej = parse_extraction({"facts": [
        ev("medications", "Энап", subject=None, dosage="", status=None, polarity=None, frequency="  "),
    ]})
    assert not rej
    f = facts[0]
    assert (f.subject, f.polarity, f.status, f.dosage, f.frequency) == ("self", "positive", None, None, None)


def test_numbers_in_text_fields_coerced():
    facts, _, rej = parse_extraction({"facts": [ev("medications", "Энап", dosage=5)]})
    assert not rej and facts[0].dosage == "5"


def test_null_required_field_still_rejected():
    _, _, rej = parse_extraction({"facts": [ev("medications", None)]})
    assert rej and "name" in rej[0].errors[0]


def test_query_classification_requires_type():
    with pytest.raises(ValidationError):
        QueryClassification(query_types=[])


def test_negation_drops_invalid_status_but_affirmation_does_not():
    facts, _, rej = parse_extraction({"facts": [
        ev("chronic_conditions", "гастрит", polarity="negative", status="confirmed_absent"),
        ev("medications", "Но-шпа", status="dont_know"),
    ]})
    assert [f.name for f in facts] == ["гастрит"] and facts[0].status is None
    assert len(rej) == 1 and "status" in rej[0].errors[0]
