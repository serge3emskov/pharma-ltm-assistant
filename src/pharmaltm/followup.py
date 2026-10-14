"""Этап 4: уточняющие вопросы после ответа.

Что нужно выяснить перед рекомендацией, решает код (блок «НУЖНО УТОЧНИТЬ» — раздел памяти без
данных, противоречие, устаревшая запись). Задала ли модель эти вопросы, тоже проверяет код:
если в ответе нет вопроса о разделе, код дописывает его к ответу — так же, как предупреждение
из проверки по справочнику. Прогоны 2 и 3 показали, что модель с ограничениями по профилю
в контексте подбирает средство и не спрашивает про лекарства и хронические заболевания.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .retrieval import ClarifyItem
from .sections import Section
from .lexicon import fold

S = Section

# Вопрос покупателю (о себе / о другом человеке) и признак того, что модель о разделе уже спросила
_QUESTIONS: dict[tuple[Section, str | None], tuple[str, str, str]] = {
    (S.MEDICATIONS, None): ("Принимаете ли вы сейчас другие лекарства?",
                            "Принимает ли {subject} сейчас другие лекарства?", r"лекарств|препарат|принима|пьет|пьёт"),
    (S.ALLERGIES, None): ("Есть ли у вас аллергия на лекарства?", "Есть ли аллергия на лекарства?", r"аллерг|непереносим"),
    (S.CHRONIC_CONDITIONS, None): ("Есть ли у вас хронические заболевания?", "Есть ли хронические заболевания?",
                                   r"хронич|заболеван|болезн|диагноз"),
    (S.SPECIAL_STATES, None): ("Нет ли беременности или кормления грудью?", "Нет ли беременности или кормления грудью?",
                               r"беремен|корми|лактац|грудн"),
    (S.DEMOGRAPHICS, "age"): ("Сколько вам лет?", "Сколько лет?", r"возраст|сколько лет|\bлет\b"),
    (S.DEMOGRAPHICS, "weight_kg"): ("Какой у вас вес?", "Какой вес?", r"\bвес|весит"),
    (S.DEMOGRAPHICS, None): ("Сколько вам лет?", "Сколько лет?", r"возраст|сколько лет|\bлет\b"),
    (S.LIFESTYLE, None): ("Водите ли вы машину, употребляете ли алкоголь?", "Есть ли что-то из образа жизни, "
                          "что важно учесть (вождение, алкоголь)?", r"рул[её]м|машин|автомобил|вожд|алкогол"),
    (S.SYMPTOMS, None): ("Какие симптомы и как давно?", "Какие симптомы и как давно?", r"симптом|давно|сколько дней"),
    (S.REPORTED_DIAGNOSES, None): ("Какой диагноз ставил врач?", "Какой диагноз ставил врач?", r"диагноз|врач постав"),
}

# Предложение — вопрос: знак вопроса или просьба уточнить
_ASK_RE = re.compile(r"\?|уточн|подскаж|скажите|сообщите|напишите|расскажите")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")


def _question(item: ClarifyItem) -> tuple[str, str, str]:
    return _QUESTIONS.get((item.section, item.key)) or _QUESTIONS[(item.section, None)]


@dataclass
class ClarifyCheck:
    """Какие вопросы из «НУЖНО УТОЧНИТЬ» модель задала, а какие дописал код."""

    answer: str
    text: str
    asked: list[ClarifyItem] = field(default_factory=list)
    appended: list[ClarifyItem] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def label(i: ClarifyItem) -> str:
            return i.section.value + (f".{i.key}" if i.key else "")
        return {"asked": [label(i) for i in self.asked], "appended": [label(i) for i in self.appended],
                "questions": self.questions}


def clarify_check(answer: str, items: Sequence[ClarifyItem], subject: str = "self",
                  recommended: bool = True) -> ClarifyCheck:
    """Дописывает к ответу вопросы из «НУЖНО УТОЧНИТЬ», которых в ответе нет.

    recommended — модель назвала препарат. Блок называется «уточнить перед рекомендацией»: если
    препарата в ответе нет (покупатель рассказал о себе, модель только спрашивает), код молчит.
    """
    if not recommended:
        items = []
    asks = [fold(s) for s in _SENTENCE_RE.findall(answer) if _ASK_RE.search(fold(s))]
    asked, appended, questions = [], [], []
    for item in items:
        own, other, pattern = _question(item)
        if any(re.search(pattern, s) for s in asks):
            asked.append(item)
            continue
        appended.append(item)
        q = own if subject == "self" else other.format(subject=subject)
        if q not in questions:
            questions.append(q)
    text = answer
    if questions:
        who = "" if subject == "self" else f" (для: {subject})"
        lines = [f"❓ Чтобы подобрать средство безопасно, уточните{who}:"] + [f"- {q}" for q in questions]
        text = answer.rstrip() + "\n\n" + "\n".join(lines)
    return ClarifyCheck(answer, text, asked, appended, questions)
