"""Предметные разделы долгосрочной памяти и их политики.

Политика раздела — это всё, что код знает о разделе без LLM:
какие статусы допустимы, что считается «актуальным», различается ли
«нет данных» и «подтверждено отсутствие», сколько живёт факт и что
делать с устаревшим фактом.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Section(str, Enum):
    MEDICATIONS = "medications"
    ALLERGIES = "allergies"
    CHRONIC_CONDITIONS = "chronic_conditions"
    SPECIAL_STATES = "special_states"
    DEMOGRAPHICS = "demographics"
    LIFESTYLE = "lifestyle"
    SYMPTOMS = "symptoms"
    REPORTED_DIAGNOSES = "reported_diagnoses"


class FactKind(str, Enum):
    ENTITY = "entity"        # ключ факта — нормализованное название (МНН, заболевание)
    ATTRIBUTE = "attribute"  # ключ факта — имя атрибута (возраст, вождение)


class StaleAction(str, Enum):
    DROP = "drop"  # устаревший факт не возвращается в контекст (остаётся в истории)
    FLAG = "flag"  # возвращается с пометкой и попадает в список «уточнить»


# Подписи статусов для контекста основной модели
STATUS_LABELS: dict[str, str] = {
    "active": "актуально",
    "prescribed": "назначено, приём ещё не начат",
    "paused": "перерыв в приёме",
    "stopped": "больше не принимает",
    "remission": "ремиссия",
    "resolved": "прошло",
    "denied": "подтверждено отсутствие",
}


# Уточнения подписей для отдельных разделов
SECTION_STATUS_LABELS: dict[str, dict[str, str]] = {
    "medications": {"active": "принимает", "denied": "не принимает"},
}

# Подписи атрибутов фактов
DATA_LABELS: dict[str, str] = {
    "dosage": "дозировка",
    "frequency": "режим",
    "indication": "показание",
    "reaction_type": "тип",
    "reaction": "реакция",
    "pregnancy_week": "неделя",
    "severity": "выраженность",
    "duration": "длительность",
    "diagnosed_by": "кто поставил",
}


DATA_VALUE_LABELS: dict[str, str] = {"allergy": "аллергия", "intolerance": "непереносимость"}


def status_label(section: "Section", status: str) -> str:
    return SECTION_STATUS_LABELS.get(section.value, {}).get(status) or STATUS_LABELS.get(status, status)


@dataclass(frozen=True)
class SectionPolicy:
    section: Section
    title: str
    kind: FactKind
    statuses: tuple[str, ...]
    active_statuses: tuple[str, ...]
    default_status: str = "active"
    # Статус, в который переходит существующий факт при отрицании
    negation_status: str = "denied"
    # Различать «нет данных» и «подтверждено отсутствие» на уровне раздела
    absence_aware: bool = False
    # Показывать основной модели факты со статусом denied («аллергии на X нет»)
    include_denied: bool = False
    # Срок актуальности в днях: общий (ключ "*") или по ключу факта
    ttl_days: dict[str, int] = field(default_factory=dict)
    on_stale: StaleAction = StaleAction.DROP
    # Атрибуты, допустимые для раздела (только для ATTRIBUTE)
    attributes: tuple[str, ...] = ()
    # Достаточно ли «нет» по одному пункту, чтобы раздел считался выясненным.
    # «Не беременна» — да; «на пенициллин аллергии нет» — нет, про другие аллергии всё ещё неизвестно.
    denied_item_answers_section: bool = False

    @property
    def inactive_statuses(self) -> tuple[str, ...]:
        return tuple(s for s in self.statuses if s not in self.active_statuses)

    def ttl_for(self, key: str) -> int | None:
        return self.ttl_days.get(key, self.ttl_days.get("*"))


POLICIES: dict[Section, SectionPolicy] = {
    Section.MEDICATIONS: SectionPolicy(
        section=Section.MEDICATIONS,
        title="Лекарства",
        kind=FactKind.ENTITY,
        statuses=("prescribed", "active", "paused", "stopped", "denied"),
        active_statuses=("prescribed", "active", "paused"),
        negation_status="stopped",
        absence_aware=True,
    ),
    Section.ALLERGIES: SectionPolicy(
        section=Section.ALLERGIES,
        title="Аллергии и непереносимость",
        kind=FactKind.ENTITY,
        statuses=("active", "denied"),
        active_statuses=("active",),
        absence_aware=True,
        include_denied=True,
    ),
    Section.CHRONIC_CONDITIONS: SectionPolicy(
        section=Section.CHRONIC_CONDITIONS,
        title="Хронические заболевания",
        kind=FactKind.ENTITY,
        statuses=("active", "remission", "denied"),
        active_statuses=("active", "remission"),
        absence_aware=True,
        include_denied=True,
    ),
    Section.SPECIAL_STATES: SectionPolicy(
        section=Section.SPECIAL_STATES,
        title="Беременность и лактация",
        kind=FactKind.ENTITY,
        statuses=("active", "denied"),
        active_statuses=("active",),
        absence_aware=True,
        include_denied=True,
        # Беременность и ГВ заканчиваются сами — давнее «да» и давнее «нет» надо переспросить
        ttl_days={"беременность": 280, "лактация": 365, "планирование беременности": 180},
        on_stale=StaleAction.FLAG,
        denied_item_answers_section=True,
    ),
    Section.DEMOGRAPHICS: SectionPolicy(
        section=Section.DEMOGRAPHICS,
        title="Возраст, пол, вес",
        kind=FactKind.ATTRIBUTE,
        statuses=("active",),
        active_statuses=("active",),
        ttl_days={"weight_kg": 180},  # вес детей быстро меняется, а от него зависит дозировка
        on_stale=StaleAction.FLAG,
        attributes=("age", "sex", "weight_kg"),
    ),
    Section.LIFESTYLE: SectionPolicy(
        section=Section.LIFESTYLE,
        title="Образ жизни",
        kind=FactKind.ATTRIBUTE,
        statuses=("active",),
        active_statuses=("active",),
        ttl_days={"*": 365},
        on_stale=StaleAction.FLAG,
        attributes=("smoking", "alcohol", "driving"),
    ),
    Section.SYMPTOMS: SectionPolicy(
        section=Section.SYMPTOMS,
        title="Жалобы и симптомы",
        kind=FactKind.ENTITY,
        statuses=("active", "resolved", "denied"),
        active_statuses=("active",),
        negation_status="resolved",
        include_denied=True,
        ttl_days={"*": 14},  # жалоба двухнедельной давности уже не про текущее обращение
        on_stale=StaleAction.DROP,
    ),
    Section.REPORTED_DIAGNOSES: SectionPolicy(
        section=Section.REPORTED_DIAGNOSES,
        title="Диагнозы со слов покупателя",
        kind=FactKind.ENTITY,
        statuses=("active", "resolved", "denied"),
        active_statuses=("active",),
        negation_status="resolved",
        ttl_days={"*": 30},
        on_stale=StaleAction.DROP,
    ),
}

ABSENCE_AWARE_SECTIONS: tuple[Section, ...] = tuple(
    s for s, p in POLICIES.items() if p.absence_aware
)

ATTRIBUTE_LABELS: dict[str, str] = {
    "age": "возраст",
    "sex": "пол",
    "weight_kg": "вес, кг",
    "smoking": "курение",
    "alcohol": "алкоголь",
    "driving": "водит автомобиль",
}
