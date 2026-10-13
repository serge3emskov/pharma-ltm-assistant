"""Baseline-классификатор запроса на регулярных выражениях.

Нужен, чтобы read-path работал без GPU, и как нижняя планка для
сравнения с LLM-классификатором на этапе оценки.
"""
from __future__ import annotations

import re

from .schemas import QueryClassification, QueryType

_RULES: list[tuple[QueryType, re.Pattern]] = [
    (QueryType.PROFILE_REVIEW, re.compile(
        r"что ты (\w+ ){0,2}(знаешь|помнишь|запомнил)|мой профиль|что (у тебя )?записано обо мне")),
    (QueryType.THERAPY_HISTORY, re.compile(
        r"(раньше|прежде|до этого|в прошлом).{0,30}(принимал|пил|назнача|лечил)|что я (принимал|пил)")),
    (QueryType.PREGNANCY_LACTATION, re.compile(r"беремен|кормящ|кормлю|грудн\w* вскармлив|\bгв\b|лактац")),
    (QueryType.DRUG_COMPATIBILITY, re.compile(
        r"совмест|сочета|взаимодейств|вместе с|одновременно|с алкогол|после алкогол|можно ли .{1,40} с ")),
    (QueryType.ANALOG, re.compile(r"аналог|замен|дешевл|дженерик|вместо")),
    (QueryType.DOSING, re.compile(
        r"как (принимать|пить|давать|применять)|сколько (раз|таблет|давать|пить|мг|капель)|дозиров|\bдоз[ауы]\b")),
    (QueryType.SYMPTOM_RELIEF, re.compile(
        r"\bот (головн|голов|температур|кашл|насморк|бол|горл|изжог|аллерги|диаре|понос|запор|простуд|тошнот)"
        r"|что (выпить|принять|посовет|взять|купить)|посоветуй|болит|заложен|температур")),
]

_SUBJECT_RE: list[tuple[str, re.Pattern]] = [
    ("ребёнок", re.compile(r"ребен|ребён|малыш")),
    ("сын", re.compile(r"\bсын")),
    ("дочь", re.compile(r"\bдоч")),
    ("мама", re.compile(r"\bмам[еаыу]?\b|\bматер")),
    ("папа", re.compile(r"\bпап[еаыу]?\b|\bотц")),
    ("муж", re.compile(r"\bмуж[уа]?\b")),
    ("жена", re.compile(r"\bжен[еуы]\b|\bжена\b")),
]


def classify_query_rules(text: str) -> QueryClassification:
    t = text.lower().replace("ё", "е")
    types = [qt for qt, rx in _RULES if rx.search(t)]
    subject = next((s for s, rx in _SUBJECT_RE if rx.search(t)), "self")
    return QueryClassification(query_types=types or [QueryType.GENERAL], subject=subject)
