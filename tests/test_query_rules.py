"""Baseline-классификатор запроса (нижняя планка для LLM-классификатора)."""
import pytest

from pharmaltm import QueryType as Q, classify_query_rules


@pytest.mark.parametrize("text,expected", [
    ("Что выпить от головной боли?", {Q.SYMPTOM_RELIEF}),
    ("Можно ли нурофен вместе с конкором?", {Q.DRUG_COMPATIBILITY}),
    ("Есть аналог Нольпазы подешевле?", {Q.ANALOG}),
    ("Как давать сироп ребёнку, сколько раз в день?", {Q.DOSING}),
    ("Что я раньше принимал от давления?", {Q.THERAPY_HISTORY}),
    ("Что ты обо мне помнишь?", {Q.PROFILE_REVIEW}),
    ("Во сколько вы закрываетесь?", {Q.GENERAL}),
])
def test_query_types(text, expected):
    assert set(classify_query_rules(text).query_types) == expected


def test_multi_label():
    got = set(classify_query_rules("Можно ли кормящей маме ибупрофен от головной боли?").query_types)
    assert {Q.PREGNANCY_LACTATION, Q.SYMPTOM_RELIEF} <= got


@pytest.mark.parametrize("text,subject", [
    ("что дать ребёнку от температуры", "ребёнок"),
    ("маме нужно что-то от давления", "мама"),
    ("что выпить от изжоги", "self"),
])
def test_subject(text, subject):
    assert classify_query_rules(text).subject == subject
