"""Проверка безопасности в коде: противопоказания и взаимодействия по справочнику.

Принцип тот же, что у памяти: LLM пишет ответ, а то, что можно проверить
детерминированно, проверяет код.

* Профиль — из памяти: принимаемые препараты, аллергии, хронические заболевания,
  беременность и ГВ, возраст, образ жизни. Только сведения со слов покупателя.
* Препарат → действующие вещества и коды АТХ (DrugCatalog) → классы (safety_data/classes.json).
* Правила (safety_data/rules.json) сопоставляют классы с профилем и с другими препаратами.

Проверка работает в двух местах:

1. До ответа: препараты из вопроса покупателя проверяются, а ограничения по профилю
   («НПВП — противопоказано: язвенная болезнь») попадают в контекст основной модели.
2. После ответа: препараты, которые назвала модель, проверяются ещё раз. Если ответ не
   называет причину предупреждения, предупреждение дописывается к ответу кодом.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import reference_data as ref
from .catalog import DrugCatalog, DrugInfo
from .lexicon import clean_name, fold
from .reference_data import CHILD_SUBJECTS
from .sections import POLICIES, Section
from .store import Fact, MemoryStore

Level = Literal["info", "caution", "avoid", "contraindicated"]
LEVELS: tuple[str, ...] = ("info", "caution", "avoid", "contraindicated")
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}
LEVEL_LABELS = {
    "contraindicated": "противопоказано",
    "avoid": "не рекомендуется без врача",
    "caution": "с осторожностью",
    "info": "к сведению",
}
DATA_DIR = Path(__file__).resolve().parent / "safety_data"

STATE_LABELS = {"беременность": "беременность", "лактация": "грудное вскармливание",
                "планирование беременности": "планирование беременности"}
LIFESTYLE_LABELS = {"driving": "вождение автомобиля", "alcohol": "употребление алкоголя", "smoking": "курение"}
STATUS_SUFFIX = {"active": "уже принимается", "prescribed": "назначен врачом", "paused": "перерыв в приёме"}


# =========================================================================== правила


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DrugClass(_Model):
    id: str
    label: str
    atc: list[str] = Field(default_factory=list)
    inn: list[str] = Field(default_factory=list)
    exclude_atc: list[str] = Field(default_factory=list)
    # То же по путям введения из лекарственных форм ГРЛС (в выгрузке нет кодов АТХ)
    exclude_routes: list[Literal["systemic", "skin", "eye", "ear", "nasal", "vaginal", "mucosal", "inhal"]] = \
        Field(default_factory=list)
    allergy_keys: list[str] = Field(default_factory=list)
    cross_allergy: Level | None = None

    def contains(self, components: Iterable[str], atc: Sequence[str], routes: Sequence[str] = ()) -> bool:
        if atc and self.exclude_atc and all(any(c.startswith(p) for p in self.exclude_atc) for c in atc):
            return False  # все формы препарата местные (гель, капли в глаза и т. п.)
        if routes and self.exclude_routes and set(routes) <= set(self.exclude_routes):
            return False
        if any(c.startswith(p) for c in atc for p in self.atc):
            return True
        return bool(set(self.inn) & set(components))


class Selector(_Model):
    classes: list[str] = Field(default_factory=list)
    inn: list[str] = Field(default_factory=list)
    any: bool = False

    @model_validator(mode="after")
    def _not_empty(self) -> "Selector":
        if not (self.classes or self.inn or self.any):
            raise ValueError("пустой селектор: нужен classes, inn или any")
        return self

    def matches(self, drug: "SafetyDrug") -> bool:
        return self.any or bool(set(self.classes) & drug.classes) or bool(set(self.inn) & set(drug.components))


class Rule(_Model):
    id: str
    kind: Literal["drug_drug", "condition", "special_state", "age", "lifestyle", "allergy_cross"]
    drug: Selector
    other: Selector | None = None
    allergen: Selector | None = None
    conditions: list[str] = Field(default_factory=list)
    states: list[Literal["беременность", "лактация", "планирование беременности"]] = Field(default_factory=list)
    age_below: int | None = None
    age_from: int | None = None
    attribute: Literal["smoking", "alcohol", "driving"] | None = None
    values: list[Literal["да", "нет", "иногда"]] = Field(default_factory=lambda: ["да", "иногда"])
    level: Level
    message: str
    reason: str
    source: str = ""
    # Более общие правила, которые это правило заменяет для того же препарата и той же причины:
    # «нимесулид при беременности» заменяет «НПВП при беременности»
    replaces: list[str] = Field(default_factory=list)
    # Показывать в «ограничениях по профилю» до ответа. False — для рецептурных групп (фторхинолоны, статины):
    # ассистент их не предлагает, а подсказка о них только удлиняет контекст. Для препарата, названного
    # в вопросе или в ответе, правило работает всегда.
    hint: bool = True

    @model_validator(mode="after")
    def _kind_fields(self) -> "Rule":
        need = {
            "drug_drug": self.other is not None,
            "condition": bool(self.conditions),
            "special_state": bool(self.states),
            "age": self.age_below is not None or self.age_from is not None,
            "lifestyle": self.attribute is not None,
            "allergy_cross": self.allergen is not None,
        }[self.kind]
        if not need:
            raise ValueError(f"{self.id}: не заполнены поля для правила типа {self.kind}")
        re.compile(self.reason)
        return self


class BuiltinRule(_Model):
    level: Level
    message: str
    reason: str


class DuplicateRule(_Model):
    level: Level
    avoid: list[str] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)
    message: str
    reason: str
    replaces: list[str] = Field(default_factory=list)


class RuleSet(_Model):
    classes: list[DrugClass]
    rules: list[Rule]
    builtin: dict[Literal["allergy_direct", "allergy_class", "allergy_cross", "intolerance", "age_unknown",
                          "rx_only"], BuiltinRule]
    duplicate_ingredient: DuplicateRule

    @model_validator(mode="after")
    def _references(self) -> "RuleSet":
        ids = [c.id for c in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("повторяющиеся id классов")
        rule_ids = [r.id for r in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("повторяющиеся id правил")
        known = set(ids)
        for r in self.rules:
            for sel in (r.drug, r.other, r.allergen):
                if sel is not None and set(sel.classes) - known:
                    raise ValueError(f"{r.id}: неизвестные классы {sorted(set(sel.classes) - known)}")
            if set(r.replaces) - set(rule_ids):
                raise ValueError(f"{r.id}: replaces ссылается на неизвестные правила {sorted(set(r.replaces) - set(rule_ids))}")
        if set(self.duplicate_ingredient.replaces) - set(rule_ids):
            raise ValueError("duplicate_ingredient.replaces: неизвестные правила")
        return self

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "RuleSet":
        d = Path(directory) if directory else DATA_DIR
        classes = json.loads((d / "classes.json").read_text(encoding="utf-8"))
        rules = json.loads((d / "rules.json").read_text(encoding="utf-8"))
        return cls(classes=classes["classes"], rules=rules["rules"], builtin=rules["builtin"],
                   duplicate_ingredient=rules["duplicate_ingredient"])

    def class_label(self, class_id: str) -> str:
        return next((c.label for c in self.classes if c.id == class_id), class_id)


# =========================================================================== препарат и профиль


@dataclass(frozen=True)
class SafetyDrug:
    info: DrugInfo
    classes: frozenset[str]
    origin: Literal["memory", "question", "prescribed", "answer", "allergy"]
    status: str | None = None

    @property
    def canonical(self) -> str:
        return self.info.canonical

    @property
    def components(self) -> tuple[str, ...]:
        return self.info.components

    @property
    def label(self) -> str:
        return self.info.label

    def as_dict(self) -> dict[str, Any]:
        return {"raw": self.info.raw, "canonical": self.canonical, "components": list(self.components),
                "atc": list(self.info.atc), "classes": sorted(self.classes), "origin": self.origin,
                "matched": self.info.matched}


@dataclass(frozen=True)
class ProfileAllergy:
    key: str
    intolerance: bool
    class_ids: frozenset[str]          # аллергия на группу целиком: «пенициллины»
    drug: SafetyDrug | None = None     # аллергия на препарат: «Амоксиклав»

    @property
    def label(self) -> str:
        return ("непереносимость: " if self.intolerance else "аллергия: ") + self.key


@dataclass
class Profile:
    subject: str
    medications: list[SafetyDrug] = field(default_factory=list)
    allergies: list[ProfileAllergy] = field(default_factory=list)
    conditions: dict[str, str] = field(default_factory=dict)   # ключ → статус
    states: dict[str, bool] = field(default_factory=dict)      # беременность → давно не подтверждалось
    age: int | None = None
    sex: str | None = None
    lifestyle: dict[str, str] = field(default_factory=dict)
    child: bool = False
    known_drugs: set[str] = field(default_factory=set)  # все препараты из памяти, в том числе бывшие и аллергены

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "medications": [m.label for m in self.medications],
            "allergies": [a.label for a in self.allergies], "conditions": self.conditions,
            "states": sorted(self.states), "age": self.age, "sex": self.sex, "lifestyle": self.lifestyle,
            "child": self.child,
        }


# =========================================================================== результат проверки


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class Alert:
    rule_id: str
    level: str
    drug: SafetyDrug
    trigger_kind: Literal["drug", "duplicate", "condition", "state", "age", "lifestyle", "allergy", "dispensing"]
    trigger_key: str
    trigger: str
    message: str
    reason: str
    other: SafetyDrug | None = None
    order: int = 0
    via: tuple[str, ...] = ()  # компоненты комбинированного препарата, из-за которых сработало правило

    def _pair_label(self) -> str:
        assert self.other is not None
        suffix = f" ({STATUS_SUFFIX.get(self.other.status or '', 'уже принимается')})" \
            if self.other.origin == "memory" else ""
        return f"{self.drug.label} + {self.other.label}{suffix}"

    def render(self) -> str:
        return _cap(self._render())

    def _render(self) -> str:
        if self.trigger_kind == "dispensing":
            return f"{self.drug.label} — {self.message}."
        level = LEVEL_LABELS[self.level]
        via = f"компонент: {', '.join(self.via)}" if self.via else ""
        if self.other is not None:
            return f"{self._pair_label()} — {level}{f' ({via})' if via else ''}: {self.message}."
        trigger = f"{self.trigger}; {via}" if via else self.trigger
        return f"{self.drug.label} — {level} ({trigger}): {self.message}."

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule_id, "level": self.level, "drug": self.drug.canonical, "drug_raw": self.drug.info.raw,
                "trigger": self.trigger, "trigger_kind": self.trigger_kind, "via": list(self.via),
                "other": self.other.canonical if self.other else None, "text": self.render()}


@dataclass(frozen=True)
class Restriction:
    rule_id: str
    level: str
    target: str
    trigger: str
    message: str
    note: str = ""  # пояснение для сгруппированного вида: «детям до 12 лет» вместо «противопоказан детям до 12 лет»

    def render(self) -> str:
        return _cap(f"{self.target} — {LEVEL_LABELS[self.level]} ({self.trigger}): {self.message}")

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule_id, "level": self.level, "target": self.target, "trigger": self.trigger,
                "text": self.render()}


_NOTE_LEVEL_RE = re.compile(r"^(?:[—–-]\s*)?(?:противопоказан\w*|не рекомендуется\w*)?\s*")
_NOTE_STATE_RE = re.compile(r"^при (?:беременности|грудном вскармливании)(?: и (?:грудном вскармливании|беременности))?\s*")


def _short_note(message: str) -> str:
    """Пояснение без того, что уже сказано уровнем и причиной группы.

    «противопоказан детям до 12 лет» → «детям до 12 лет»; «при грудном вскармливании противопоказаны
    (на время лечения кормление прекращают)» → «на время лечения кормление прекращают».
    """
    note = _NOTE_LEVEL_RE.sub("", message.strip())
    note = _NOTE_LEVEL_RE.sub("", _NOTE_STATE_RE.sub("", note))
    if note.startswith("(") and note.endswith(")"):
        note = note[1:-1]
    return note.replace(";", ",").strip()


def render_restrictions(restrictions: Sequence[Restriction]) -> list[str]:
    """Ограничения одной причины — одной строкой: «Возраст 6 — противопоказано: кеторолак — …; нимесулид — …»."""
    groups: dict[tuple[str, str], list[Restriction]] = {}
    for r in restrictions:
        groups.setdefault((r.trigger, r.level), []).append(r)
    lines = []
    for (trigger, level), rs in groups.items():
        if len(rs) == 1:
            lines.append(f"- {rs[0].render()}")
        else:
            items = "; ".join(f"{r.target} — {r.note}" if r.note else r.target for r in rs)
            lines.append(f"- {_cap(trigger)} — {LEVEL_LABELS[level]}: {items}")
    return lines


SAFETY_HEADER = ("### ПРОВЕРКА ПО СПРАВОЧНИКУ (выполнена кодом по инструкциям к препаратам). Не предлагай препараты "
                 "с пометкой «противопоказано» или «не рекомендуется без врача»; если покупатель спрашивает о таком "
                 "препарате, прямо назови риск и причину")


@dataclass
class SafetyReport:
    """Что код знает до ответа: профиль, предупреждения по препаратам из вопроса, ограничения по профилю."""

    profile: Profile
    question_drugs: list[SafetyDrug] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    restrictions: list[Restriction] = field(default_factory=list)

    def render(self) -> list[str]:
        if not self.alerts and not self.restrictions:
            return []
        lines = ["", SAFETY_HEADER]
        if self.alerts:
            lines.append("Препараты из вопроса:")
            lines += [f"- {a.render()}" for a in self.alerts]
        if self.restrictions:
            lines.append("Ограничения по профилю (чего не предлагать):")
            lines += render_restrictions(self.restrictions)
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile.as_dict(), "question_drugs": [d.as_dict() for d in self.question_drugs],
                "alerts": [a.as_dict() for a in self.alerts],
                "restrictions": [r.as_dict() for r in self.restrictions]}


@dataclass
class PostCheck:
    """Проверка готового ответа: какие предупреждения модель учла, а какие дописал код."""

    raw: str
    text: str
    answer_drugs: list[SafetyDrug]
    alerts: list[Alert]
    appended: list[Alert]   # не учтено моделью — дописано кодом
    covered: list[Alert]    # модель назвала причину
    declined: list[Alert] = field(default_factory=list)  # модель отговорила от препарата, причину не назвала
    doses: list[str] = field(default_factory=list)       # дозы и кратность в ответе (нарушение правила промпта)

    def as_dict(self) -> dict[str, Any]:
        return {"answer_drugs": [d.label for d in self.answer_drugs], "doses": self.doses,
                "alerts": [a.as_dict() for a in self.alerts],
                "appended": [a.as_dict() for a in self.appended],
                "covered": [a.as_dict() for a in self.covered],
                "declined": [a.as_dict() for a in self.declined]}


_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
# Дозы и кратность в ответе: промпт их запрещает (источник доз — инструкция), код считает нарушения
DOSE_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:мг|мкг|мл|г|ед|таблет\w*|капсул\w*|капел\w*)(?![а-яё])"
                     r"|\d+\s*(?:-|–)?\s*\d*\s*раз\w*\s+в\s+(?:день|сутки)", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text) if s.strip()]


# =========================================================================== проверка


class SafetyChecker:
    def __init__(self, catalog: DrugCatalog | None = None, rules: RuleSet | None = None):
        self.catalog = catalog or DrugCatalog()
        self.rules = rules or RuleSet.load()
        self._classes = {c.id: c for c in self.rules.classes}
        self._allergen_names = {fold(e.canonical) for e in ref.ALLERGENS}
        self._replaces: dict[str, set[str]] = {r.id: set(r.replaces) for r in self.rules.rules if r.replaces}
        self._replaces["duplicate_ingredient"] = set(self.rules.duplicate_ingredient.replaces)

    # ------------------------------------------------------------------ препараты
    def classes_for(self, components: Iterable[str], atc: Sequence[str], routes: Sequence[str] = ()) -> frozenset[str]:
        comps = list(components)
        return frozenset(c.id for c in self.rules.classes if c.contains(comps, atc, routes))

    def drug(self, info: DrugInfo, origin: str, status: str | None = None) -> SafetyDrug:
        return SafetyDrug(info, self.classes_for(info.components, info.atc, info.routes), origin,  # type: ignore[arg-type]
                          status)

    def resolve(self, name: str, origin: str = "question") -> SafetyDrug:
        return self.drug(self.catalog.resolve(name), origin)

    def mentions(self, text: str, origin: str) -> list[SafetyDrug]:
        return [self.drug(info, origin) for info in self.catalog.mentions(text)]

    # ------------------------------------------------------------------ профиль из памяти
    def _allergy(self, key: str, raw_names: Sequence[str], intolerance: bool) -> ProfileAllergy:
        k = fold(key)
        class_ids = frozenset(c.id for c in self.rules.classes if k in {fold(x) for x in c.allergy_keys})
        drug = None
        if not class_ids and k not in self._allergen_names:
            info = self.catalog.from_key(key, raw_names)
            if info.matched:
                drug = self.drug(info, "allergy")
        return ProfileAllergy(key, intolerance, class_ids, drug)

    def profile(self, store: MemoryStore, user_id: str, subject: str = "self",
                now: datetime | None = None) -> Profile:
        """Профиль субъекта: только актуальные сведения со слов покупателя."""
        now = now or datetime.now(timezone.utc)
        prof = Profile(subject=subject, child=subject in CHILD_SUBJECTS)
        for f in store.list_facts(user_id, subject):
            policy = POLICIES[f.section]
            if f.section == Section.MEDICATIONS:
                prof.known_drugs.add(f.key)
                if f.status in policy.active_statuses:
                    info = self.catalog.from_key(f.key, f.raw_names)
                    prof.medications.append(self.drug(info, "memory", f.status))
            elif f.section == Section.ALLERGIES and f.status == "active":
                prof.known_drugs.add(f.key)
                prof.allergies.append(self._allergy(f.key, f.raw_names, f.data.get("reaction_type") == "intolerance"))
            elif f.section == Section.CHRONIC_CONDITIONS and f.status in policy.active_statuses:
                prof.conditions[f.key] = f.status
            elif f.section == Section.SPECIAL_STATES and f.status == "active":
                ttl = policy.ttl_for(f.key)
                prof.states[f.key] = ttl is not None and now - f.last_confirmed_at > timedelta(days=ttl)
            elif f.section == Section.DEMOGRAPHICS:
                value = f.data.get("value")
                if f.key == "age" and isinstance(value, (int, float)):
                    prof.age = int(value)
                elif f.key == "sex":
                    prof.sex = str(value)
            elif f.section == Section.LIFESTYLE and f.data.get("value"):
                prof.lifestyle[f.key] = str(f.data["value"])
        if prof.age is not None:
            prof.child = prof.age < 18
        return prof

    # ------------------------------------------------------------------ правила для одного препарата
    def _alert(self, rule_id: str, level: str, drug: SafetyDrug, kind: str, key: str, trigger: str,
               message: str, reason: str, other: SafetyDrug | None = None, order: int = 0,
               sel: Selector | None = None) -> Alert:
        via = self._via(sel, drug) if sel is not None else ()
        return Alert(rule_id, level, drug, kind, key, trigger, message, reason, other, order, via)  # type: ignore[arg-type]

    def _via(self, sel: Selector, d: SafetyDrug) -> tuple[str, ...]:
        """Какие компоненты комбинированного препарата попали под правило («Терафлю» → фенирамин)."""
        if len(d.components) < 2 or sel.any:
            return ()
        inn = set(sel.inn)
        for cid in sel.classes:
            inn |= set(self._classes[cid].inn)
        return tuple(c for c in d.components if c in inn)

    def _allergy_alerts(self, d: SafetyDrug, profile: Profile) -> list[Alert]:
        b = self.rules.builtin
        out: list[Alert] = []
        for a in profile.allergies:
            key = f"allergy:{a.key}"
            hits: list[tuple[str, str, str, str]] = []  # rule_id, level, message, reason
            same = a.drug is not None and (a.drug.canonical == d.canonical
                                           or set(a.drug.components) & set(d.components))
            if same or fold(a.key) in {fold(c) for c in d.components}:
                hits.append(("allergy_direct", b["allergy_direct"].level, b["allergy_direct"].message,
                             b["allergy_direct"].reason))
            if a.class_ids & d.classes:
                hits.append(("allergy_class", b["allergy_class"].level, b["allergy_class"].message,
                             b["allergy_class"].reason))
            if a.drug is not None:
                for cid in a.drug.classes & d.classes:
                    level = self._classes[cid].cross_allergy
                    if level:
                        hits.append(("allergy_cross", level, b["allergy_cross"].message, b["allergy_cross"].reason))
            allergen_classes = set(a.class_ids) | (set(a.drug.classes) if a.drug else set())
            allergen_inn = set(a.drug.components) if a.drug else {fold(a.key)}
            for r in self.rules.rules:
                if r.kind != "allergy_cross" or not r.drug.matches(d):
                    continue
                assert r.allergen is not None
                if set(r.allergen.classes) & allergen_classes or set(r.allergen.inn) & allergen_inn:
                    hits.append((r.id, r.level, r.message, r.reason))
            for rule_id, level, message, reason in hits:
                if a.intolerance:
                    # Непереносимость (тошнота, боль в желудке) — не аллергия: строже «не без врача» не ставим
                    cap = b["intolerance"].level
                    level = level if LEVEL_RANK[level] < LEVEL_RANK[cap] else cap
                    rule_id, message, reason = "intolerance", b["intolerance"].message, b["intolerance"].reason
                out.append(self._alert(rule_id, level, d, "allergy", key, a.label, message, reason, order=-2))
        return out

    def _profile_alerts(self, d: SafetyDrug, profile: Profile) -> list[Alert]:
        out: list[Alert] = []
        for i, r in enumerate(self.rules.rules):
            if r.kind in ("drug_drug", "allergy_cross") or not r.drug.matches(d):
                continue
            if r.kind == "condition":
                for c in r.conditions:
                    if c in profile.conditions:
                        trig = c + (" (ремиссия)" if profile.conditions[c] == "remission" else "")
                        out.append(self._alert(r.id, r.level, d, "condition", f"condition:{c}", trig,
                                               r.message, r.reason, order=i, sel=r.drug))
            elif r.kind == "special_state":
                for s in r.states:
                    if s in profile.states:
                        trig = STATE_LABELS[s] + (" (давно не подтверждалось)" if profile.states[s] else "")
                        out.append(self._alert(r.id, r.level, d, "state", f"state:{s}", trig, r.message, r.reason,
                                               order=i, sel=r.drug))
            elif r.kind == "age":
                if profile.age is not None:
                    if ((r.age_below is not None and profile.age < r.age_below)
                            or (r.age_from is not None and profile.age >= r.age_from)):
                        out.append(self._alert(r.id, r.level, d, "age", "age", f"возраст {profile.age}",
                                               r.message, r.reason, order=i, sel=r.drug))
                elif profile.child and r.age_below is not None:
                    au = self.rules.builtin["age_unknown"]
                    out.append(self._alert(f"age_unknown:{r.id}", au.level, d, "age", "age", "возраст не указан",
                                           au.message.format(detail=r.message), au.reason, order=i, sel=r.drug))
            elif r.kind == "lifestyle":
                assert r.attribute is not None
                value = profile.lifestyle.get(r.attribute)
                if value in r.values and (r.age_from is None or profile.age is None or profile.age >= r.age_from):
                    out.append(self._alert(r.id, r.level, d, "lifestyle", f"lifestyle:{r.attribute}",
                                           LIFESTYLE_LABELS[r.attribute], r.message, r.reason, order=i, sel=r.drug))
        return out

    def _same_product(self, d: SafetyDrug, o: SafetyDrug) -> bool:
        if d.canonical != o.canonical:
            return False
        # Один и тот же препарат, названный дважды, — не дублирование.
        # Другое торговое наименование того же МНН из памяти — дублирование («Нурофен» при приёме «МИГ»).
        return o.origin != "memory" or clean_name(d.info.raw) == clean_name(o.info.raw) \
            or clean_name(d.info.raw) == clean_name(d.canonical)

    def _pair_alerts(self, d: SafetyDrug, o: SafetyDrug) -> list[Alert]:
        if self._same_product(d, o):
            return []
        out: list[Alert] = []
        key = f"drug:{o.canonical}"
        dup = self.rules.duplicate_ingredient
        shared = [c for c in d.components if c in o.components and c not in dup.ignore]
        if shared:
            level = "avoid" if any(c in dup.avoid for c in shared) else dup.level
            comp = shared[0]
            reason = dup.reason.format(component_pattern=re.escape(fold(comp)[:6]))
            out.append(self._alert("duplicate_ingredient", level, d, "duplicate", key, o.label,
                                   dup.message.format(component=", ".join(shared)), reason, o, order=-1))
        for i, r in enumerate(self.rules.rules):
            if r.kind != "drug_drug":
                continue
            assert r.other is not None
            if r.drug.matches(d) and r.other.matches(o):
                out.append(self._alert(r.id, r.level, d, "drug", key, o.label, r.message, r.reason, o, i, r.drug))
            elif r.drug.matches(o) and r.other.matches(d):
                out.append(self._alert(r.id, r.level, d, "drug", key, o.label, r.message, r.reason, o, i, r.other))
        return out

    def _dedupe(self, alerts: list[Alert]) -> list[Alert]:
        """Убирает повторы.

        * Аллергия и возраст: на пару «препарат — причина» одно предупреждение, самое строгое.
        * Остальное: разные правила для одной причины остаются (у «Терафлю» при беременности два компонента
          с разными противопоказаниями), кроме явно заменённых общих правил (replaces).
        """
        best: dict[tuple[str, ...], Alert] = {}
        for a in alerts:
            k = (a.drug.canonical, a.trigger_key) if a.trigger_kind in ("allergy", "age") \
                else (a.drug.canonical, a.trigger_key, a.rule_id)
            cur = best.get(k)
            if cur is None or (LEVEL_RANK[a.level], -a.order) > (LEVEL_RANK[cur.level], -cur.order):
                best[k] = a
        kept = list(best.values())
        replaced: set[tuple[str, str, str]] = set()
        for a in kept:
            for rid in self._replaces.get(a.rule_id, ()):
                replaced.add((a.drug.canonical, a.trigger_key, rid))
        return [a for a in kept if (a.drug.canonical, a.trigger_key, a.rule_id) not in replaced]

    def check(self, drugs: Sequence[SafetyDrug], profile: Profile, pairs: bool = True) -> list[Alert]:
        """Предупреждения для препаратов-кандидатов: с профилем и между собой.

        Пары двух препаратов из ответа модели не проверяются: «парацетамол или ибупрофен» — это
        варианты на выбор, а не совместный приём. pairs=False отключает и пары препаратов из вопроса.
        """
        cands: list[SafetyDrug] = []
        for d in drugs:
            if d.info.matched and all(d.canonical != c.canonical for c in cands):
                cands.append(d)
        found: list[Alert] = []
        rx = self.rules.builtin["rx_only"]
        # Препараты из памяти (принимает, принимал, аллергия) — назначал врач, о рецепте не напоминаем
        known = {m.canonical for m in profile.medications} | profile.known_drugs | {
            a.drug.canonical for a in profile.allergies if a.drug is not None}
        for i, d in enumerate(cands):
            found += self._allergy_alerts(d, profile)
            found += self._profile_alerts(d, profile)
            # Только по рецепту (по ГРЛС): ассистент такой препарат не предлагает, а покупателю нужен рецепт.
            # Препарат, который покупатель уже принимает, — назначен врачом, о нём не напоминаем
            if d.info.otc is False and d.canonical not in known and d.origin != "prescribed":
                found.append(self._alert("rx_only", rx.level, d, "dispensing", "dispensing", "рецептурный препарат",
                                         rx.message, rx.reason, order=-3))
            for o in profile.medications:
                found += self._pair_alerts(d, o)
            for o in cands[i + 1:]:
                if o.origin == d.origin == "answer" or (not pairs and o.origin == d.origin):
                    continue
                found += self._pair_alerts(d, o)
        position = {c.canonical: i for i, c in enumerate(cands)}
        return sorted(self._dedupe(found), key=lambda a: (position[a.drug.canonical], -LEVEL_RANK[a.level], a.order))

    # ------------------------------------------------------------------ ограничения по профилю
    def selector_label(self, sel: Selector) -> str:
        parts = [self.rules.class_label(c) for c in sel.classes] + list(sel.inn)
        return ", ".join(parts) if parts else "любые препараты"

    def restrictions(self, profile: Profile, min_level: str = "avoid", limit: int = 10) -> list[Restriction]:
        """Чего не предлагать этому покупателю, ещё до того, как модель назовёт препарат."""
        out: list[Restriction] = []

        def add(rule_id: str, level: str, target: str, trigger: str, message: str, note: str | None = None) -> None:
            if LEVEL_RANK[level] >= LEVEL_RANK[min_level]:
                out.append(Restriction(rule_id, level, target, trigger, message,
                                       _short_note(message) if note is None else note))

        for a in profile.allergies:
            b = self.rules.builtin
            for cid in a.class_ids:
                add("allergy_class", b["allergy_class"].level, self.rules.class_label(cid), a.label,
                    b["allergy_class"].message, "")
            if a.drug is not None:
                comps = ", ".join(a.drug.components)
                kind = "intolerance" if a.intolerance else "allergy_direct"
                add(kind, b[kind].level, f"препараты, содержащие {comps}", a.label, b[kind].message, "")
                for cid in a.drug.classes:
                    cl = self._classes[cid].cross_allergy
                    if cl:
                        if a.intolerance:
                            cap = b["intolerance"].level
                            add("intolerance", cl if LEVEL_RANK[cl] < LEVEL_RANK[cap] else cap,
                                f"другие {self.rules.class_label(cid)}", a.label, b["intolerance"].message)
                        else:
                            add("allergy_cross", cl, f"другие {self.rules.class_label(cid)}", a.label,
                                b["allergy_cross"].message, "возможна перекрёстная реакция")
            allergen_classes = set(a.class_ids) | (set(a.drug.classes) if a.drug else set())
            for r in self.rules.rules:
                if r.kind == "allergy_cross" and r.allergen and set(r.allergen.classes) & allergen_classes:
                    add(r.id, r.level, self.selector_label(r.drug), a.label, r.message)

        for r in self.rules.rules:
            if not r.hint:
                continue
            target = self.selector_label(r.drug)
            if r.kind == "condition":
                for c in r.conditions:
                    if c in profile.conditions:
                        add(r.id, r.level, target, c, r.message)
            elif r.kind == "special_state":
                for s in r.states:
                    if s in profile.states:
                        add(r.id, r.level, target, STATE_LABELS[s], r.message)
            elif r.kind == "age" and profile.age is not None:
                if ((r.age_below is not None and profile.age < r.age_below)
                        or (r.age_from is not None and profile.age >= r.age_from)):
                    add(r.id, r.level, target, f"возраст {profile.age}", r.message)
            elif r.kind == "lifestyle" and r.attribute:
                if profile.lifestyle.get(r.attribute) in r.values and (
                        r.age_from is None or profile.age is None or profile.age >= r.age_from):
                    add(r.id, r.level, target, LIFESTYLE_LABELS[r.attribute], r.message)
            elif r.kind == "drug_drug" and r.other is not None and not r.drug.any:
                # Правила записаны так, что drug — то, что могут купить, other — то, что уже принимают
                for m in profile.medications:
                    if r.other.matches(m):
                        add(r.id, r.level, target, f"принимает {m.label}", r.message)

        best: dict[tuple[str, str], Restriction] = {}
        for x in out:
            k = (x.target, x.trigger)
            if k not in best or LEVEL_RANK[x.level] > LEVEL_RANK[best[k].level]:
                best[k] = x
        ranked = sorted(best.values(), key=lambda x: -LEVEL_RANK[x.level])
        return ranked[:limit]

    # ------------------------------------------------------------------ до ответа
    _ALTERNATIVES_RE = re.compile(r"\bили\b|что лучше|что выбрать|чем отличается|какой из")
    _TOGETHER_RE = re.compile(r"вместе|одновремен|совмест|сочета|параллельн|запива|\bплюс\b|\+|после него|после нее")

    def is_combination(self, text: str) -> bool:
        """Спрашивают о совместном приёме, а не о выборе одного из препаратов («Нурофен или Найз?»)."""
        t = fold(text)
        return bool(self._TOGETHER_RE.search(t)) or not self._ALTERNATIVES_RE.search(t)

    _PRESCRIBED_RE = re.compile(r"выписал|выписали|назначил|назначили|назначен|по рецепту|рецепт")
    _ANALOG_RE = re.compile(r"замен|вместо|аналог|альтернатив|подешевле|дешевле|что-то другое|что то другое")

    def report(self, profile: Profile, question_text: str | None, with_restrictions: bool = True,
               max_restrictions: int = 6, choosing: bool | None = None) -> SafetyReport:
        """Проверка до ответа.

        Ограничения по профилю нужны, когда модель будет выбирать препарат сама: «что дать от
        температуры?» или «чем заменить Флемоксин?» (choosing; если не задано — по словам вопроса).
        Если покупатель спрашивает о конкретном препарате, в контекст идут только предупреждения
        о нём: список «чего не предлагать» там только отвлекает модель.
        """
        origin = "prescribed" if question_text and self._PRESCRIBED_RE.search(fold(question_text)) else "question"
        q_drugs = self.mentions(question_text, origin) if question_text else []
        alerts = self.check(q_drugs, profile, pairs=self.is_combination(question_text or ""))
        if choosing is None:
            choosing = bool(question_text and self._ANALOG_RE.search(fold(question_text)))
        restrictions: list[Restriction] = []
        if with_restrictions and (not q_drugs or choosing):
            # То, что уже сказано про препараты из вопроса, второй раз не повторяем
            # (кроме аллергий на группу: «другие НПВП» полезны, даже если про Найз из вопроса уже предупредили)
            said = {(a.rule_id, f"принимает {a.other.label}" if a.other is not None else a.trigger)
                    for a in alerts
                    if a.trigger_kind != "allergy" or a.rule_id in ("allergy_direct", "intolerance")}
            restrictions = [r for r in self.restrictions(profile, limit=50)
                            if (r.rule_id, r.trigger) not in said][:max_restrictions]
        return SafetyReport(profile, q_drugs, alerts, restrictions)

    # ------------------------------------------------------------------ после ответа
    _PROHIBIT_RE = re.compile(r"нельзя|не рекоменд|противопоказ|избега|не принима|не прием|не дава|не подход|"
                              r"не подойд|не стоит|не использ|не применя|откаж|исключ|лучше не|не следует|"
                              r"не нужно|запрещ")
    _WARN_RE = re.compile(_PROHIBIT_RE.pattern + r"|осторожн|риск|опасн|только по назначению|без врача|"
                                                r"посоветуйтесь с врачом|обратитесь к врачу|согласуйте с врачом|"
                                                r"вызыва|вызв|привод|привест|повыша|усилива|ослабля|снижа|вред|"
                                                r"превыш|передозир|уже принима")
    _PERMIT_RE = re.compile(r"\bможно\b|подойд[её]т|подходит|рекомендую|возьмите|примите|принимайте|допуст")
    # «не противопоказан», «противопоказаний нет» — это разрешение, а не запрет
    _NEGATED_PROHIBIT_RE = re.compile(r"\bне\s+(?:является\s+|был[аио]?\s+)?(?:противопоказ|запрещ)\w*|"
                                      r"\b(?:нет|без)\s+противопоказан\w*|противопоказаний\s+нет")

    @staticmethod
    def _stems(d: SafetyDrug) -> set[str]:
        """Основы названия препарата для поиска в тексте: «Панадол» → «панад», «парацетамол» → «парацетам»."""
        names = [clean_name(d.info.raw)]
        if len(d.components) == 1:
            names.append(clean_name(d.canonical))
        stems = set()
        for n in names:
            first = n.split()[0] if n else ""
            if len(first) >= 5:
                stems.add(first[: max(5, len(first) - 2)])
        return stems

    def _coverage(self, alert: Alert, answer: str, sentences: list[tuple[str, set[str]]],
                  single_drug: bool = False) -> str:
        """covered — причина названа; declined — модель отговаривает от препарата без причины; missing.

        single_drug — в ответе речь об одном препарате: тогда к нему относится любое предложение.
        """
        pattern = re.compile(alert.reason)
        drugs = [alert.drug] + ([alert.other] if alert.other else [])
        names = {d.canonical for d in drugs}
        stems = set().union(*(self._stems(d) for d in drugs))
        # «Не противопоказан» заменяем на «можно»: такое предложение — разрешение. Строгое предупреждение
        # оно не закрывает, даже если в нём названа причина: «при аденоме Супрастин не противопоказан»
        endorsed = {i for i, (s, _) in enumerate(sentences) if self._NEGATED_PROHIBIT_RE.search(s)}
        strict = LEVEL_RANK[alert.level] >= LEVEL_RANK["avoid"]
        sentences = [(self._NEGATED_PROHIBIT_RE.sub(" можно ", s), found) for s, found in sentences]
        # Где препарат назван: найден справочником или по основе названия («парацетамола»)
        named = [i for i, (s, found) in enumerate(sentences) if found & names or any(st in s for st in stems)]
        drug_stems = self._stems(alert.drug)
        named_drug = [i for i in named
                      if alert.drug.canonical in sentences[i][1] or any(st in sentences[i][0] for st in drug_stems)]
        # К нему относятся и следующее предложение без других препаратов («Супрастин помогает при
        # аллергии. Учитывая, что вы за рулём, — он вызывает сонливость»), а если в ответе один препарат —
        # весь ответ
        scope = set(range(len(sentences))) if single_drug and named else set(named)
        for i in named:
            if i + 1 < len(sentences) and not (sentences[i + 1][1] - names):
                scope.add(i + 1)
        if scope:
            texts = [sentences[i][0] for i in sorted(scope) if not (strict and i in endorsed)]
            # Чем строже предупреждение, тем определённее должна быть формулировка:
            # «противопоказано» — запрет («Нурофен при язве нельзя»), а не «НПВП могут усугубить язву»;
            # «не рекомендуется без врача» — запрет или прямое указание на риск/осторожность;
            # «с осторожностью» — достаточно назвать причину
            marker = {"contraindicated": self._PROHIBIT_RE, "avoid": self._WARN_RE}.get(alert.level)
            if any(pattern.search(s) and (marker is None or marker.search(s)) for s in texts):
                return "covered"
            # Отговаривает от препарата — только в предложении, где назван он сам (а не то, что уже
            # принимают): «Аспирин детям нельзя»; «принимаете Кардиомагнил — избегайте…» про Нурофен не говорит
            for i in named_drug:
                s = sentences[i][0]
                if self._PROHIBIT_RE.search(s) and not self._PERMIT_RE.search(self._PROHIBIT_RE.sub(" ", s)):
                    return "declined"
            return "missing"
        return "covered" if pattern.search(fold(answer)) else "missing"

    @staticmethod
    def _inn_named(d: SafetyDrug) -> bool:
        """Назван по МНН, а не торговым наименованием: «левотироксин» для «левотироксин натрия» тоже."""
        raw = clean_name(d.info.raw)
        names = {clean_name(d.canonical)} | {clean_name(c) for c in d.components}
        return any(n == raw or n.startswith(raw + " ") for n in names)

    def _recommended(self, answer_drugs: list[SafetyDrug], question_drugs: Sequence[SafetyDrug],
                     profile: Profile) -> list[SafetyDrug]:
        """Убирает из препаратов ответа упоминания-пояснения, которые не являются новым препаратом.

        «Терафлю содержит парацетамол, фенилэфрин и фенирамин», «Кардиомагнил (ацетилсалициловая
        кислота)», «вы принимаете левотироксин» (а в памяти — Эутирокс): МНН, названное как состав
        другого препарата из диалога или как действующее вещество того, что покупатель уже принимает.
        """
        others = [*question_drugs, *(d for d in answer_drugs if not self._inn_named(d)), *profile.medications]
        out = []
        for d in answer_drugs:
            inn = self._inn_named(d)
            if inn and any(o.canonical != d.canonical and set(d.components) <= set(o.components) for o in others):
                continue  # состав другого препарата
            if any(m.canonical == d.canonical and (inn or clean_name(m.info.raw) == clean_name(d.info.raw))
                   for m in profile.medications):
                continue  # то, что покупатель уже принимает (тем же названием или по МНН): назначение врача
            out.append(d)
        return out

    def post_check(self, answer: str, profile: Profile, question_drugs: Sequence[SafetyDrug] = (),
                   min_level: str = "caution") -> PostCheck:
        """Проверка ответа модели. Неучтённые предупреждения уровня min_level и выше дописываются к ответу.

        Предупреждение считается учтённым, если в предложении, где назван препарат, есть причина
        (регулярное выражение reason правила) или модель прямо отговаривает от препарата («Аспирин
        детям нельзя»). Если препарат в ответе не назван (покупатель спросил о нём, а модель предложила
        другое), достаточно, чтобы причина была где-то в ответе.
        Эвристика намеренно строгая: лучше лишнее предупреждение, чем пропущенное.
        """
        answer_drugs = self._recommended(self.mentions(answer, "answer"), question_drugs, profile)
        # Препараты из вопроса — первыми: при совпадении МНН препарат считается названным покупателем
        alerts = self.check([*question_drugs, *answer_drugs], profile)
        sentences = [(fold(s), {i.canonical for i in self.catalog.mentions(s)}) for s in split_sentences(answer)]
        single = len({d.canonical for d in [*question_drugs, *answer_drugs]}) == 1
        groups: dict[str, list[Alert]] = {"covered": [], "declined": [], "missing": []}
        for a in alerts:
            if LEVEL_RANK[a.level] >= LEVEL_RANK[min_level]:
                groups[self._coverage(a, answer, sentences, single and a.other is None)].append(a)
        appended, covered = groups["missing"], groups["covered"]
        text = answer
        if appended:
            who = "" if profile.subject == "self" else f" (для: {profile.subject})"
            lines = [f"⚠ Проверка по справочнику{who}:"] + [f"- {a.render()}" for a in appended]
            lines.append("Проверка автоматическая и не заменяет инструкцию и консультацию фармацевта.")
            text = answer.rstrip() + "\n\n" + "\n".join(lines)
        doses = [m.group(0) for m in DOSE_RE.finditer(answer)]
        return PostCheck(answer, text, answer_drugs, alerts, appended, covered, groups["declined"], doses)
