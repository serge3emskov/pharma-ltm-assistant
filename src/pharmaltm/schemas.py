"""Контракт между LLM и кодом.

Extractor (LLM) возвращает JSON. Каждое событие валидируется отдельно:
невалидное событие в память не попадает, но и не блокирует остальные.
В схемах нет ID, времени и типа изменения — их определяет код.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

ShortText = Annotated[str, Field(min_length=1, max_length=120)]
Evidence = Annotated[str, Field(min_length=1, max_length=500)]
Polarity = Literal["positive", "negative"]


class _Strict(BaseModel):
    # Числа в текстовых полях («dosage»: 5) приводим к строке — это не меняет смысл
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, coerce_numbers_to_str=True)

    @model_validator(mode="before")
    @classmethod
    def _null_means_absent(cls, data: Any) -> Any:
        """null и пустая строка от модели означают «поле не указано»: срабатывают значения по умолчанию."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None and not (isinstance(v, str) and not v.strip())}
        return data


# --------------------------------------------------------------------------- факты


def _literal_values(annotation: Any) -> set:
    values: set = set()
    for arg in get_args(annotation) or (annotation,):
        if get_origin(arg) is Literal:
            values |= set(get_args(arg))
    return values


class _EntityEvent(_Strict):
    subject: ShortText = "self"   # для кого: сам покупатель, ребёнок, мама...
    name: ShortText               # как сказал пользователь, без нормализации
    polarity: Polarity = "positive"
    evidence: Evidence            # фрагмент реплики, из которого извлечён факт

    @model_validator(mode="before")
    @classmethod
    def _negation_drops_bad_status(cls, data: Any) -> Any:
        """«Гастрита нет» со статусом "confirmed_absent": смысл несёт отрицание, лишний статус отбрасываем.
        Для утверждения недопустимый статус по-прежнему ошибка — там он меняет смысл."""
        if isinstance(data, dict) and data.get("polarity") == "negative" and "status" in cls.model_fields:
            if data.get("status") is not None and data["status"] not in _literal_values(
                    cls.model_fields["status"].annotation):
                data = {k: v for k, v in data.items() if k != "status"}
        return data


class MedicationEvent(_EntityEvent):
    section: Literal["medications"]
    status: Literal["prescribed", "active", "paused", "stopped"] | None = None
    dosage: ShortText | None = None
    frequency: ShortText | None = None
    indication: ShortText | None = None


class AllergyEvent(_EntityEvent):
    section: Literal["allergies"]
    reaction_type: Literal["allergy", "intolerance"] | None = None
    reaction: ShortText | None = None


class ChronicConditionEvent(_EntityEvent):
    section: Literal["chronic_conditions"]
    status: Literal["active", "remission"] | None = None


_SPECIAL_STATE_SYNONYMS = {
    "беременность": "беременность",
    "беременна": "беременность",
    "в положении": "беременность",
    "лактация": "лактация",
    "кормлю грудью": "лактация",
    "кормящая": "лактация",
    "грудное вскармливание": "лактация",
    "гв": "лактация",
    "планирование беременности": "планирование беременности",
    "планирую беременность": "планирование беременности",
}


class SpecialStateEvent(_EntityEvent):
    section: Literal["special_states"]
    name: Literal["беременность", "лактация", "планирование беременности"]
    pregnancy_week: int | None = Field(default=None, ge=1, le=42)

    @field_validator("name", mode="before")
    @classmethod
    def _canonical_name(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().lower().replace("ё", "е")
            return _SPECIAL_STATE_SYNONYMS.get(key, v)
        return v

    @model_validator(mode="after")
    def _week_only_for_pregnancy(self) -> "SpecialStateEvent":
        if self.pregnancy_week is not None and self.name != "беременность":
            raise ValueError("pregnancy_week допустим только для «беременность»")
        return self


class SymptomEvent(_EntityEvent):
    section: Literal["symptoms"]
    status: Literal["active", "resolved"] | None = None
    severity: Literal["лёгкая", "умеренная", "сильная"] | None = None
    duration: ShortText | None = None


class DiagnosisEvent(_EntityEvent):
    section: Literal["reported_diagnoses"]
    status: Literal["active", "resolved"] | None = None
    diagnosed_by: Literal["врач", "сам"] | None = None


_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SEX = {
    "м": "м", "муж": "м", "мужской": "м", "мужчина": "м", "male": "м", "m": "м",
    "ж": "ж", "жен": "ж", "женский": "ж", "женщина": "ж", "female": "ж", "f": "ж",
}


class DemographicsEvent(_Strict):
    section: Literal["demographics"]
    subject: ShortText = "self"
    attribute: Literal["age", "sex", "weight_kg"]
    value: int | float | str
    evidence: Evidence

    @model_validator(mode="after")
    def _check_value(self) -> "DemographicsEvent":
        if self.attribute == "sex":
            key = str(self.value).strip().lower()
            if key not in _SEX:
                raise ValueError(f"недопустимое значение пола: {self.value!r}")
            self.value = _SEX[key]
            return self
        num = self.value if isinstance(self.value, (int, float)) else None
        if num is None:
            m = _NUM_RE.search(str(self.value))
            if not m:
                raise ValueError(f"{self.attribute}: ожидается число, получено {self.value!r}")
            num = float(m.group().replace(",", "."))
        if self.attribute == "age":
            if not 0 <= num <= 120:
                raise ValueError(f"возраст вне диапазона 0–120: {num}")
            self.value = int(num)
        else:
            if not 0.5 <= num <= 300:
                raise ValueError(f"вес вне диапазона 0.5–300 кг: {num}")
            self.value = round(float(num), 1)
        return self


class LifestyleEvent(_Strict):
    section: Literal["lifestyle"]
    subject: ShortText = "self"
    attribute: Literal["smoking", "alcohol", "driving"]
    value: Literal["да", "нет", "иногда"]
    details: ShortText | None = None
    evidence: Evidence


FactEvent = Annotated[
    Union[
        MedicationEvent,
        AllergyEvent,
        ChronicConditionEvent,
        SpecialStateEvent,
        DemographicsEvent,
        LifestyleEvent,
        SymptomEvent,
        DiagnosisEvent,
    ],
    Field(discriminator="section"),
]

EntityEvent = Union[
    MedicationEvent, AllergyEvent, ChronicConditionEvent, SpecialStateEvent,
    SymptomEvent, DiagnosisEvent,
]
AttributeEvent = Union[DemographicsEvent, LifestyleEvent]

# Поля события, которые становятся атрибутами факта
FACT_DATA_FIELDS: dict[str, tuple[str, ...]] = {
    "medications": ("dosage", "frequency", "indication"),
    "allergies": ("reaction_type", "reaction"),
    "chronic_conditions": (),
    "special_states": ("pregnancy_week",),
    "symptoms": ("severity", "duration"),
    "reported_diagnoses": ("diagnosed_by",),
    "demographics": ("value",),
    "lifestyle": ("value", "details"),
}


# ---------------------------------------------------------- статусы разделов


class SectionStatusEvent(_Strict):
    """«Аллергий нет» / «не знаю» — сведения о разделе целиком."""

    section: Literal["medications", "allergies", "chronic_conditions", "special_states"]
    subject: ShortText = "self"
    status: Literal["confirmed_absent", "dont_know"]
    evidence: Evidence


# ------------------------------------------------------------ разбор ответа LLM


class Rejection(BaseModel):
    kind: Literal["fact", "section_status", "output"]
    raw: Any
    errors: list[str]


_FACT_ADAPTER = TypeAdapter(FactEvent)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _format_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append(f"{loc}: {err['msg']}")
    return out


def parse_extraction(
    raw: str | dict,
) -> tuple[list[Any], list[SectionStatusEvent], list[Rejection]]:
    """Разбирает ответ extractor-а: `{"facts": [...], "section_statuses": [...]}`.

    Возвращает валидные факты, валидные статусы разделов и отклонённые элементы.
    Ошибка в одном элементе не отменяет остальные.
    """
    if isinstance(raw, str):
        text = _FENCE_RE.sub("", raw.strip())
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            return [], [], [Rejection(kind="output", raw=text[:2000], errors=[f"JSON: {exc}"])]
    if not isinstance(raw, dict):
        return [], [], [Rejection(kind="output", raw=raw, errors=["ожидается JSON-объект"])]

    unknown_keys = set(raw) - {"facts", "section_statuses"}
    rejections: list[Rejection] = []
    if unknown_keys:
        rejections.append(
            Rejection(kind="output", raw=sorted(unknown_keys), errors=["неизвестные ключи верхнего уровня"])
        )

    facts: list[Any] = []
    for item in raw.get("facts") or []:
        try:
            facts.append(_FACT_ADAPTER.validate_python(item))
        except ValidationError as exc:
            rejections.append(Rejection(kind="fact", raw=item, errors=_format_errors(exc)))

    statuses: list[SectionStatusEvent] = []
    for item in raw.get("section_statuses") or []:
        try:
            statuses.append(SectionStatusEvent.model_validate(item))
        except ValidationError as exc:
            rejections.append(Rejection(kind="section_status", raw=item, errors=_format_errors(exc)))

    return facts, statuses, rejections


# ------------------------------------------------------------ классификация запроса


class QueryType(str, Enum):
    SYMPTOM_RELIEF = "symptom_relief"            # подобрать средство от симптома
    DRUG_COMPATIBILITY = "drug_compatibility"    # можно ли вместе, с алкоголем
    DOSING = "dosing"                            # как принимать, сколько давать
    ANALOG = "analog"                            # замена, аналог, дешевле
    PREGNANCY_LACTATION = "pregnancy_lactation"  # можно ли беременным / при ГВ
    THERAPY_HISTORY = "therapy_history"          # что принимал раньше
    PROFILE_REVIEW = "profile_review"            # что ты обо мне знаешь
    GENERAL = "general"                          # без персонального контекста


class QueryClassification(_Strict):
    """Контракт classifier-а запроса (на этапе 2 его заполнит LLM)."""

    query_types: list[QueryType] = Field(min_length=1)
    subject: ShortText = "self"
