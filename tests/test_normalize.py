"""Нормализация названий: торговое наименование → МНН, синонимы → канон, опечатки."""
import pytest

from pharmaltm.lexicon import LexEntry, Lexicon, clean_name, parse_dosage
from pharmaltm.sections import Section


@pytest.mark.parametrize("raw,expected,method", [
    ("Энап 5 мг", "эналаприл", "exact"),
    ("эналаприл", "эналаприл", "exact"),
    ("Нурофен Экспресс Форте", "ибупрофен", "prefix"),
    ("Конкор Кор 2,5 мг", "бисопролол", "exact"),
    ("Но-шпа", "дротаверин", "exact"),
    ("L-тироксин 50 мкг", "левотироксин натрия", "exact"),
    ("нурафен", "ибупрофен", "fuzzy"),
    ("эналоприл", "эналаприл", "fuzzy"),
])
def test_drug_names(normalizer, raw, expected, method):
    r = normalizer.entity(Section.MEDICATIONS, raw)
    assert (r.normalized, r.method) == (expected, method)


def test_combination_product_keeps_components(normalizer):
    r = normalizer.entity(Section.MEDICATIONS, "ТераФлю")
    assert r.components == ("парацетамол", "фенилэфрин", "фенирамин")
    assert "анальгетики-антипиретики" in r.classes


def test_unknown_drug_is_unmatched_not_guessed(normalizer):
    r = normalizer.entity(Section.MEDICATIONS, "витамин Д3 2000 МЕ")
    assert r.method == "unmatched" and r.normalized == "витамин д3"


def test_ambiguous_fuzzy_match_is_not_guessed():
    lex = Lexicon([LexEntry("препарат-а", ("абвгдеж",)), LexEntry("препарат-б", ("абвгдез",))])
    r = lex.lookup("абвгдеи")
    assert r.method == "unmatched" and set(r.candidates) == {"препарат-а", "препарат-б"}


def test_short_names_are_never_fuzzy_matched(normalizer):
    assert normalizer.entity(Section.MEDICATIONS, "миг").normalized == "ибупрофен"   # точное — можно
    assert normalizer.entity(Section.MEDICATIONS, "мик").method == "unmatched"       # нечёткое — нет


@pytest.mark.parametrize("raw,expected", [
    ("Энап 5 мг", "5 мг"),
    ("Конкор 2,5мг", "2,5 мг"),
    ("Амоксиклав 875/125 мг", "875/125 мг"),
    ("Эутирокс 50 мкг", "50 мкг"),
    ("Кардиомагнил 75", None),
])
def test_dosage_parsed_by_code(raw, expected):
    assert parse_dosage(raw) == expected


def test_clean_name_removes_form_and_numbers():
    assert clean_name("Нурофен таблетки 200 мг №10") == "нурофен"


@pytest.mark.parametrize("section,raw,expected", [
    (Section.ALLERGIES, "пенициллин", "пенициллины"),
    (Section.ALLERGIES, "аспирин", "ацетилсалициловая кислота"),
    (Section.CHRONIC_CONDITIONS, "давление", "артериальная гипертензия"),
    (Section.CHRONIC_CONDITIONS, "диабет 2 типа", "сахарный диабет 2 типа"),
    (Section.SYMPTOMS, "голова болит", "головная боль"),
    (Section.REPORTED_DIAGNOSES, "простуда", "ОРВИ"),
])
def test_other_sections(normalizer, section, raw, expected):
    assert normalizer.entity(section, raw).normalized == expected


@pytest.mark.parametrize("raw,expected", [
    (None, "self"), ("я", "self"), ("дочке", "дочь"), ("Маме", "мама"), ("ребёнку", "ребёнок"),
    ("соседу", "соседу"),
])
def test_subjects(normalizer, raw, expected):
    assert normalizer.subject(raw) == expected
