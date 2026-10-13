"""Read-path: тип запроса → набор разделов → факты с подходящими статусами → контекст.

Без semantic top-K и без LLM-reranking: выборка предсказуема и тестируется
как обычная функция. Сюда же вынесены правила «забывания» (срок актуальности
по типу факта) и список того, что ассистенту нужно уточнить у покупателя.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Literal

from .lexicon import NormResult
from .normalize import Normalizer
from .reference_data import CHILD_SUBJECTS, MALE_SUBJECTS
from .schemas import QueryType
from .sections import ATTRIBUTE_LABELS, DATA_LABELS, DATA_VALUE_LABELS, POLICIES, FactKind, Section, StaleAction, status_label
from .store import Fact, MemoryStore, SectionState

S = Section

QUERY_SECTIONS: dict[QueryType, tuple[Section, ...]] = {
    QueryType.SYMPTOM_RELIEF: (S.SYMPTOMS, S.REPORTED_DIAGNOSES, S.MEDICATIONS, S.ALLERGIES,
                               S.CHRONIC_CONDITIONS, S.SPECIAL_STATES, S.DEMOGRAPHICS, S.LIFESTYLE),
    QueryType.DRUG_COMPATIBILITY: (S.MEDICATIONS, S.ALLERGIES, S.CHRONIC_CONDITIONS, S.SPECIAL_STATES,
                                   S.LIFESTYLE),
    QueryType.DOSING: (S.DEMOGRAPHICS, S.MEDICATIONS, S.CHRONIC_CONDITIONS, S.SPECIAL_STATES),
    QueryType.ANALOG: (S.MEDICATIONS, S.ALLERGIES),
    QueryType.PREGNANCY_LACTATION: (S.SPECIAL_STATES, S.MEDICATIONS, S.ALLERGIES, S.CHRONIC_CONDITIONS),
    QueryType.THERAPY_HISTORY: (S.MEDICATIONS, S.REPORTED_DIAGNOSES, S.CHRONIC_CONDITIONS),
    QueryType.PROFILE_REVIEW: tuple(Section),
    QueryType.GENERAL: (),
}

# Разделы, без которых рекомендацию давать рано: если о них ничего не известно — уточнить
CLARIFY_SECTIONS: dict[QueryType, tuple[Section, ...]] = {
    QueryType.SYMPTOM_RELIEF: (S.ALLERGIES, S.MEDICATIONS, S.CHRONIC_CONDITIONS, S.SPECIAL_STATES),
    QueryType.DRUG_COMPATIBILITY: (S.MEDICATIONS,),
    QueryType.DOSING: (S.SPECIAL_STATES,),
    QueryType.ANALOG: (S.ALLERGIES,),
}

# Обязательные атрибуты для запроса; weight_kg — только для детей
REQUIRED_ATTRIBUTES: dict[QueryType, tuple[str, ...]] = {
    QueryType.DOSING: ("age",),
}

# Запросы, для которых нужны и неактуальные записи
HISTORY_SECTIONS: dict[QueryType, tuple[Section, ...]] = {
    QueryType.THERAPY_HISTORY: (S.MEDICATIONS, S.REPORTED_DIAGNOSES, S.CHRONIC_CONDITIONS),
}

ClarifyReason = Literal["unknown", "conflict", "stale", "missing_attribute"]


@dataclass(frozen=True)
class ClarifyItem:
    section: Section
    reason: ClarifyReason
    key: str | None = None

    def render(self) -> str:
        title = POLICIES[self.section].title
        what = f"{title} → {ATTRIBUTE_LABELS.get(self.key, self.key)}" if self.key else title
        reason = {
            "unknown": "нет данных",
            "conflict": "противоречие: покупатель сказал «нет», но в памяти есть актуальные записи",
            "stale": "давно не подтверждалось",
            "missing_attribute": "нет данных",
        }[self.reason]
        return f"- {what}: {reason}"


@dataclass(frozen=True)
class ContextFact:
    fact: Fact
    stale: bool = False


@dataclass(frozen=True)
class PurchaseRecord:
    """Документированный источник: чек по карте лояльности."""

    date: str
    name_raw: str
    quantity: int = 1


def select_sections(query_types: Iterable[QueryType]) -> list[Section]:
    """Объединение разделов без дублей, порядок — по первому появлению."""
    out: list[Section] = []
    for qt in query_types:
        for s in QUERY_SECTIONS[qt]:
            if s not in out:
                out.append(s)
    return out


def statuses_for(section: Section, query_types: Iterable[QueryType]) -> set[str]:
    """Какие статусы фактов раздела нужны для данного набора запросов."""
    policy = POLICIES[section]
    qts = set(query_types)
    if QueryType.PROFILE_REVIEW in qts:
        return set(policy.statuses)
    allowed = set(policy.active_statuses)
    if policy.include_denied:
        allowed.add("denied")
    for qt in qts:
        if section in HISTORY_SECTIONS.get(qt, ()):
            allowed |= {s for s in policy.inactive_statuses if s != "denied"}
    return allowed


def _special_states_irrelevant(subject: str, demographics: dict[str, object]) -> bool:
    """Упрощённое правило: про беременность не спрашиваем мужчин и детей."""
    if subject in MALE_SUBJECTS or subject in CHILD_SUBJECTS:
        return True
    if demographics.get("sex") == "м":
        return True
    age = demographics.get("age")
    return isinstance(age, int) and (age < 12 or age > 55)


@dataclass
class ContextBundle:
    user_id: str
    subject: str
    query_types: list[QueryType]
    sections: list[Section]
    facts: dict[Section, list[ContextFact]] = field(default_factory=dict)
    section_states: dict[Section, SectionState] = field(default_factory=dict)
    clarify: list[ClarifyItem] = field(default_factory=list)
    purchases: list[tuple[PurchaseRecord, str]] = field(default_factory=list)
    reference: list[NormResult] = field(default_factory=list)
    now: datetime | None = None

    # ---------------------------------------------------------------- для тестов и оценки
    def keys(self, section: Section | None = None) -> set[str]:
        secs = [section] if section else list(self.facts)
        return {cf.fact.key for s in secs for cf in self.facts.get(s, [])}

    def sections_with_facts(self) -> set[Section]:
        return {s for s, items in self.facts.items() if items}

    # ---------------------------------------------------------------- рендер для основной модели
    def render(self) -> str:
        if not self.sections:
            lines = ["### ПАМЯТЬ: для этого запроса персональный контекст не нужен"]
            lines += self._render_reference()
            return "\n".join(lines)
        subj = "сам покупатель" if self.subject == "self" else self.subject
        lines = [
            "### ПАМЯТЬ АССИСТЕНТА — со слов покупателя (документами не подтверждено)",
            f"Субъект: {subj}",
        ]
        for section in self.sections:
            block = self._render_section(section)
            if block:
                lines += block
        if self.purchases:
            lines += ["", "### ДОКУМЕНТИРОВАНО — история покупок по карте лояльности"]
            for rec, normalized in self.purchases:
                lines.append(f"- {rec.date}: «{rec.name_raw}» × {rec.quantity} → {normalized}")
            lines.append("Если покупки расходятся со словами покупателя, не выбирай одну версию сам — уточни.")
        lines += self._render_reference()
        if self.clarify:
            lines += ["", "### НУЖНО УТОЧНИТЬ ПЕРЕД РЕКОМЕНДАЦИЕЙ (этого нет в памяти; "
                          "если покупатель уже сказал об этом в текущем диалоге — не переспрашивай)"]
            lines += [c.render() for c in self.clarify]
        return "\n".join(lines)

    def _render_reference(self) -> list[str]:
        if not self.reference:
            return []
        lines = ["", "### СПРАВОЧНИК — препараты, упомянутые в диалоге"]
        for m in self.reference:
            if len(m.components) > 1:
                parts = [f"- «{m.raw}» → комбинированный препарат: " + ", ".join(m.components)]
            else:
                parts = [f"- «{m.raw}» → {m.normalized}"]
            if m.classes:
                parts.append("группа: " + ", ".join(m.classes))
            lines.append("; ".join(parts))
        return lines

    def _render_section(self, section: Section) -> list[str]:
        policy = POLICIES[section]
        items = self.facts.get(section, [])
        state = self.section_states.get(section)
        lines: list[str] = []
        if state and state.state == "confirmed_absent" and not items:
            lines.append("- подтверждено: нет")
        elif state and state.state == "dont_know":
            lines.append("- покупатель сказал, что не знает")
        for cf in items:
            lines.append(_render_fact(cf, policy.kind))
        if not lines:
            return []
        return ["", f"#### {policy.title}", *lines]


def _render_fact(cf: ContextFact, kind: FactKind) -> str:
    f = cf.fact
    stale = " ⚠ давно не подтверждалось" if cf.stale else ""
    confirmed = f.last_confirmed_at.date().isoformat()
    if kind == FactKind.ATTRIBUTE:
        label = ATTRIBUTE_LABELS.get(f.key, f.key)
        details = f" ({f.data['details']})" if f.data.get("details") else ""
        return f"- {label}: {f.data.get('value')}{details} [на {confirmed}]{stale}"
    if f.status == "denied":
        # «НЕТ» заглавными: маленькая модель читала «беременность: подтверждено отсутствие» как беременность
        return f"- {f.key}: НЕТ (подтверждено отсутствие) [на {confirmed}]{stale}"
    parts = [f"- {f.key} [{status_label(f.section, f.status)}]"]
    attrs = {k: v for k, v in f.data.items() if v not in (None, "")}
    if attrs:
        parts.append("; ".join(
            f"{DATA_LABELS.get(k, k)}: {DATA_VALUE_LABELS.get(str(v), v)}" for k, v in attrs.items()
        ))
    if f.raw_names and [n.lower() for n in f.raw_names] != [f.key.lower()]:
        parts.append("со слов: " + ", ".join(f"«{n}»" for n in f.raw_names))
    if f.norm.get("method") == "unmatched":
        parts.append("название не найдено в справочнике")
    elif f.norm.get("method") == "fuzzy":
        parts.append("название распознано с опечаткой")
    parts.append(f"подтверждено {confirmed}")
    return " — ".join(parts) + stale


class ContextBuilder:
    def __init__(
        self,
        store: MemoryStore,
        normalizer: Normalizer | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.store = store
        self.normalizer = normalizer or Normalizer()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def is_stale(self, fact: Fact, now: datetime) -> bool:
        ttl = POLICIES[fact.section].ttl_for(fact.key)
        return ttl is not None and now - fact.last_confirmed_at > timedelta(days=ttl)

    def build(
        self,
        user_id: str,
        query_types: Iterable[QueryType],
        subject: str = "self",
        purchases: Iterable[PurchaseRecord] | None = None,
        reference_text: str | None = None,
    ) -> ContextBundle:
        """reference_text — реплики покупателя: упомянутые в них препараты попадают в блок СПРАВОЧНИК."""
        qts = list(dict.fromkeys(query_types)) or [QueryType.GENERAL]
        subject = self.normalizer.subject(subject)
        now = self.clock()
        sections = select_sections(qts)
        profile = QueryType.PROFILE_REVIEW in qts
        bundle = ContextBundle(user_id=user_id, subject=subject, query_types=qts, sections=sections, now=now)

        all_facts = self.store.list_facts(user_id, subject, sections)
        # Демография нужна правилам уточнения, даже если сам раздел в контекст не идёт
        demographics = {
            f.key: f.data.get("value") for f in self.store.list_facts(user_id, subject, [S.DEMOGRAPHICS])
        }

        for section in sections:
            policy = POLICIES[section]
            allowed = statuses_for(section, qts)
            selected: list[ContextFact] = []
            for f in all_facts:
                if f.section != section or f.status not in allowed:
                    continue
                stale = self.is_stale(f, now)
                if stale and policy.on_stale == StaleAction.DROP and not profile:
                    continue  # «забыли» для контекста, но история осталась
                selected.append(ContextFact(f, stale))
                if stale and policy.on_stale == StaleAction.FLAG and not profile:
                    bundle.clarify.append(ClarifyItem(section, "stale", f.key))
            bundle.facts[section] = selected
            if policy.absence_aware:
                bundle.section_states[section] = self.store.get_section_state(user_id, subject, section)

        if not profile:
            self._add_clarifications(bundle, qts, all_facts, demographics)

        if purchases and S.MEDICATIONS in sections:
            for rec in purchases:
                nr = self.normalizer.entity(S.MEDICATIONS, rec.name_raw)
                label = nr.normalized if nr.matched else f"{nr.normalized} (не найдено в справочнике)"
                bundle.purchases.append((rec, label))
        if reference_text:
            bundle.reference = self.normalizer.drug_mentions(reference_text)
        return bundle

    @staticmethod
    def _section_known(section: Section, state: SectionState | None, facts: list[Fact]) -> bool:
        """Раздел выяснен, если есть «нет»/«не знаю» на уровне раздела или актуальный факт."""
        if state is not None and state.state is not None:
            return True
        policy = POLICIES[section]
        known = set(policy.active_statuses)
        if policy.denied_item_answers_section:
            known.add("denied")
        return any(f.section == section and f.status in known for f in facts)

    def _add_clarifications(
        self, bundle: ContextBundle, qts: list[QueryType], all_facts: list[Fact], demographics: dict
    ) -> None:
        seen = {(c.section, c.key) for c in bundle.clarify}

        def add(item: ClarifyItem) -> None:
            if (item.section, item.key) not in seen:
                seen.add((item.section, item.key))
                bundle.clarify.append(item)

        for qt in qts:
            for section in CLARIFY_SECTIONS.get(qt, ()):
                if section == S.SPECIAL_STATES and _special_states_irrelevant(bundle.subject, demographics):
                    continue
                state = bundle.section_states.get(section)
                if state and state.needs_clarification:
                    add(ClarifyItem(section, "conflict"))
                    continue
                if not self._section_known(section, state, all_facts):
                    add(ClarifyItem(section, "unknown"))
            required = list(REQUIRED_ATTRIBUTES.get(qt, ()))
            if qt == QueryType.DOSING and (bundle.subject in CHILD_SUBJECTS or
                                           (isinstance(demographics.get("age"), int) and demographics["age"] < 14)):
                required.append("weight_kg")
            for attr in required:
                if attr not in demographics:
                    add(ClarifyItem(S.DEMOGRAPHICS, "missing_attribute", attr))

        # Конфликты показываем всегда, если раздел попал в контекст
        for section, state in bundle.section_states.items():
            if state.needs_clarification:
                add(ClarifyItem(section, "conflict"))
