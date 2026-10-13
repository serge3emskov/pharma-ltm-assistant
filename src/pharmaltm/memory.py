"""Write-path: валидированное событие → изменение памяти.

После extraction зона ответственности LLM заканчивается. Всё, что ниже,
детерминировано: ключ факта (через нормализацию), ID, время, тип изменения,
итоговый статус и правила для разделов с подтверждённым отсутствием.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from .normalize import Normalizer
from .schemas import FACT_DATA_FIELDS, Rejection, SectionStatusEvent, parse_extraction
from .sections import POLICIES, FactKind, Section, SectionPolicy
from .store import Fact, MemoryStore, new_id


class ChangeType(str, Enum):
    FIRST_MENTION = "first_mention"              # новый факт
    DENIAL_RECORDED = "denial_recorded"          # новый факт об отсутствии: «аллергии на X нет»
    CONFIRMATION = "confirmation"                # повтор без изменений
    UPDATE = "update"                            # изменились атрибуты или статус в пределах актуальных
    DEACTIVATION = "deactivation"                # актуальный → неактуальный (перестал принимать, прошло)
    NEGATION = "negation"                        # актуальный → «подтверждено отсутствие»
    REACTIVATION = "reactivation"                # неактуальный → снова актуальный
    SECTION_CONFIRMED_ABSENT = "section_confirmed_absent"  # «аллергий нет»
    SECTION_DONT_KNOW = "section_dont_know"                # «не знаю»
    SECTION_STATE_CLEARED = "section_state_cleared"        # «нет» / «не знаю» перекрыты новым фактом
    CONFLICT = "conflict"                        # «аллергий нет», но в памяти есть актуальная аллергия


def classify_transition(
    policy: SectionPolicy, old_status: str | None, new_status: str, data_changed: bool
) -> ChangeType:
    """Тип изменения определяется только переходом статуса и сравнением атрибутов."""
    active = policy.active_statuses
    if old_status is None:
        return ChangeType.DENIAL_RECORDED if new_status == "denied" else ChangeType.FIRST_MENTION
    old_active, new_active = old_status in active, new_status in active
    if old_active and new_active:
        return ChangeType.UPDATE if (data_changed or old_status != new_status) else ChangeType.CONFIRMATION
    if old_active and not new_active:
        return ChangeType.NEGATION if new_status == "denied" else ChangeType.DEACTIVATION
    if not old_active and new_active:
        return ChangeType.REACTIVATION
    return ChangeType.UPDATE if (data_changed or old_status != new_status) else ChangeType.CONFIRMATION


@dataclass
class AppliedChange:
    change_type: ChangeType
    section: Section
    subject: str
    key: str | None
    fact_id: str | None
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    event_id: str


@dataclass
class WriteReport:
    applied: list[AppliedChange] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ch in self.applied:
            out[ch.change_type.value] = out.get(ch.change_type.value, 0) + 1
        if self.rejected:
            out["rejected"] = len(self.rejected)
        return out

    def changes(self, change_type: ChangeType) -> list[AppliedChange]:
        return [c for c in self.applied if c.change_type == change_type]

    def to_rows(self) -> list[dict[str, Any]]:
        rows = [
            {"change": c.change_type.value, "section": c.section.value, "subject": c.subject,
             "key": c.key, "before": c.before, "after": c.after}
            for c in self.applied
        ]
        rows += [{"change": "rejected", "section": None, "subject": None, "key": None,
                  "before": r.raw, "after": "; ".join(r.errors)} for r in self.rejected]
        return rows


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryWriter:
    def __init__(
        self,
        store: MemoryStore,
        normalizer: Normalizer | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.store = store
        self.normalizer = normalizer or Normalizer()
        self.clock = clock or _utcnow

    # ------------------------------------------------------------------ вход
    def write(self, user_id: str, extraction: str | dict, dialog_id: str | None = None) -> WriteReport:
        """Принимает сырой ответ extractor-а, валидирует и применяет события."""
        facts, statuses, rejections = parse_extraction(extraction)
        now = self.clock()
        report = WriteReport(rejected=rejections)
        with self.store.transaction():
            for rej in rejections:
                self.store.add_rejection(user_id, dialog_id, now, rej.kind, rej.raw, rej.errors)
            # Сначала факты, потом статусы разделов: «аллергий нет» сверяется с уже записанным
            for ev in facts:
                report.applied += self.apply_fact(user_id, ev, dialog_id, now)
            for st in statuses:
                report.applied += self.apply_section_status(user_id, st, dialog_id, now)
        return report

    # ------------------------------------------------------------------ факты
    def apply_fact(self, user_id: str, ev: Any, dialog_id: str | None, now: datetime) -> list[AppliedChange]:
        section = Section(ev.section)
        policy = POLICIES[section]
        subject = self.normalizer.subject(ev.subject)
        incoming = {f: getattr(ev, f) for f in FACT_DATA_FIELDS[section.value] if getattr(ev, f) is not None}

        if policy.kind == FactKind.ATTRIBUTE:
            key, raw_name, norm, polarity, explicit_status = ev.attribute, None, {"method": "attribute"}, "positive", None
        else:
            nr = self.normalizer.entity(section, ev.name)
            key, raw_name, norm, polarity = nr.normalized, ev.name, nr.as_dict(), ev.polarity
            explicit_status = getattr(ev, "status", None)
            if section == Section.MEDICATIONS and "dosage" not in incoming and nr.dosage:
                incoming["dosage"] = nr.dosage  # дозировку из «Энап 5 мг» достаёт код, а не LLM

        existing = self.store.get_fact(user_id, subject, section, key)
        if existing is None:
            new_status = "denied" if polarity == "negative" else (explicit_status or policy.default_status)
            data = {} if polarity == "negative" else incoming
            fact = Fact(
                fact_id=new_id(), user_id=user_id, subject=subject, section=section, key=key,
                status=new_status, data=data, raw_names=[raw_name] if raw_name else [], norm=norm,
                first_seen_at=now, updated_at=now, last_confirmed_at=now, mentions=1,
            )
            before = None
            change = classify_transition(policy, None, new_status, bool(data))
        else:
            fact = existing
            before = existing.snapshot()
            if polarity == "negative":
                new_status = (
                    existing.status if existing.status not in policy.active_statuses else policy.negation_status
                )
                merged = existing.data  # атрибуты из отрицания не сливаем
            else:
                if explicit_status:
                    new_status = explicit_status
                elif existing.status in policy.active_statuses:
                    new_status = existing.status
                else:
                    new_status = policy.default_status
                merged = {**existing.data, **incoming}
            data_changed = merged != existing.data
            change = classify_transition(policy, existing.status, new_status, data_changed)
            fact.status, fact.data = new_status, merged
            fact.mentions += 1
            fact.last_confirmed_at = now
            if change != ChangeType.CONFIRMATION:
                fact.updated_at = now
            if raw_name and raw_name not in fact.raw_names:
                fact.raw_names.append(raw_name)

        self.store.save_fact(fact)
        after = fact.snapshot()
        event_id = self.store.add_event(
            user_id=user_id, subject=subject, section=section, fact_id=fact.fact_id,
            change_type=change.value, dialog_id=dialog_id, created_at=now,
            payload=ev.model_dump(), before=before, after=after,
        )
        changes = [AppliedChange(change, section, subject, key, fact.fact_id, before, after, event_id)]
        if policy.absence_aware:
            changes += self._sync_section_state(user_id, subject, section, fact, dialog_id, now)
        return changes

    def _sync_section_state(
        self, user_id: str, subject: str, section: Section, fact: Fact, dialog_id: str | None, now: datetime
    ) -> list[AppliedChange]:
        """Новый актуальный факт перекрывает «нет»/«не знаю»; любой факт раздела снимает флаг конфликта."""
        policy = POLICIES[section]
        st = self.store.get_section_state(user_id, subject, section)
        before = {"state": st.state, "needs_clarification": st.needs_clarification}
        if fact.status in policy.active_statuses and st.state is not None:
            st.state = None
        st.needs_clarification = False
        after = {"state": st.state, "needs_clarification": st.needs_clarification}
        if after == before:
            return []
        self.store.save_section_state(st, now)
        event_id = self.store.add_event(
            user_id=user_id, subject=subject, section=section, fact_id=None,
            change_type=ChangeType.SECTION_STATE_CLEARED.value, dialog_id=dialog_id, created_at=now,
            payload={"trigger_fact": fact.key}, before=before, after=after,
        )
        return [AppliedChange(ChangeType.SECTION_STATE_CLEARED, section, subject, None, None, before, after, event_id)]

    # ------------------------------------------------------------------ статусы разделов
    def apply_section_status(
        self, user_id: str, ev: SectionStatusEvent, dialog_id: str | None, now: datetime
    ) -> list[AppliedChange]:
        section = Section(ev.section)
        policy = POLICIES[section]
        subject = self.normalizer.subject(ev.subject)
        st = self.store.get_section_state(user_id, subject, section)
        before = {"state": st.state, "needs_clarification": st.needs_clarification}
        active_keys = [f.key for f in self.store.list_facts(user_id, subject, [section], policy.active_statuses)]

        if ev.status == "confirmed_absent":
            if active_keys:
                # Не удаляем аллергию из-за одной реплики: помечаем раздел для уточнения
                st.needs_clarification = True
                change = ChangeType.CONFLICT
            else:
                change = (ChangeType.CONFIRMATION if st.state == "confirmed_absent"
                          else ChangeType.SECTION_CONFIRMED_ABSENT)
                st.state, st.needs_clarification = "confirmed_absent", False
        else:  # dont_know
            change = ChangeType.SECTION_DONT_KNOW
            if not active_keys:
                st.state = "dont_know"  # «не знаю» осторожнее, чем прежнее «нет»

        after = {"state": st.state, "needs_clarification": st.needs_clarification}
        if active_keys:
            after["active_keys"] = active_keys
        self.store.save_section_state(st, now)
        event_id = self.store.add_event(
            user_id=user_id, subject=subject, section=section, fact_id=None, change_type=change.value,
            dialog_id=dialog_id, created_at=now, payload=ev.model_dump(), before=before, after=after,
        )
        return [AppliedChange(change, section, subject, None, None, before, after, event_id)]
