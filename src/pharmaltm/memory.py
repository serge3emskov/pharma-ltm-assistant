"""Write-path: валидированное событие → изменение памяти.

После extraction зона ответственности LLM заканчивается. Всё, что ниже,
детерминировано: ключ факта (через нормализацию), ID, время, тип изменения,
итоговый статус и правила для разделов с подтверждённым отсутствием.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from rapidfuzz import fuzz

from .lexicon import clean_name, evidence_coverage, fold
from .normalize import Normalizer
from .schemas import FACT_DATA_FIELDS, Rejection, SectionStatusEvent, parse_extraction
from .sections import POLICIES, FactKind, Section, SectionPolicy
from .store import Fact, MemoryStore, new_id


CODE_RULE_PREFIX = "код: "  # так помечены события, отклонённые правилами кода, а не схемой


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
    def write(self, user_id: str, extraction: str | dict, dialog_id: str | None = None,
              dialog_text: str | None = None) -> WriteReport:
        """Принимает сырой ответ extractor-а, валидирует, проверяет согласованность и применяет события.

        dialog_text — то, что видела модель; если передан, цитата evidence каждого события
        должна найтись в нём, иначе событие считается выдуманным.
        """
        facts, statuses, rejections = parse_extraction(extraction)
        facts, statuses, by_rules = self._consistency_check(facts, statuses, dialog_text)
        rejections += by_rules
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

    # ------------------------------------------------------------------ проверки согласованности
    # Детерминированные правила поверх валидации схемы. Каждое — ответ на класс ошибок модели,
    # найденный при оценке; модель их не видит и не может обойти.
    _NOT_DRUG_SECTIONS = (Section.CHRONIC_CONDITIONS, Section.REPORTED_DIAGNOSES, Section.SYMPTOMS)
    _ALLERGY_WORDS = ("аллерг", "непереносим")
    GENERIC_ALLERGENS = frozenset({
        "лекарства", "лекарство", "лекарственные препараты", "препараты", "медикаменты", "таблетки",
        "ничего", "ни на что", "что-либо", "что либо", "любые лекарства",
    })
    MIN_EVIDENCE_COVERAGE = 0.5
    # Слово-маркер атрибута должно быть в цитате или в вопросе ассистента («Он курит?» — «Да, много»)
    LIFESTYLE_MARKERS = {
        "smoking": re.compile(r"кур[июяеё]|курен|сигарет|табак|вейп|кальян"),
        "alcohol": re.compile(r"алког|спиртн|вин[оау]|пив[оау]|водк|коньяк|выпива|бокал|рюмк"),
        "driving": re.compile(r"рул[ьяеюё]|вожу|водит|вожден|машин|автомоб|такси|автобус|шоф[её]р|дальнобой"),
    }

    @staticmethod
    def _reject(ev: Any, reason: str) -> Rejection:
        raw = ev.model_dump() if hasattr(ev, "model_dump") else ev
        return Rejection(kind="fact", raw=raw, errors=[f"{CODE_RULE_PREFIX}{reason}"])

    def _consistency_check(
        self, facts: list[Any], statuses: list[SectionStatusEvent], dialog_text: str | None
    ) -> tuple[list[Any], list[SectionStatusEvent], list[Rejection]]:
        n = self.normalizer
        kept: list[Any] = []
        statuses = list(statuses)
        rejected: list[Rejection] = []

        if dialog_text:
            grounded: list[Any] = []
            for ev in facts:
                if evidence_coverage(ev.evidence, dialog_text) < self.MIN_EVIDENCE_COVERAGE:
                    rejected.append(self._reject(ev, f"цитата «{ev.evidence}» не найдена в диалоге"))
                else:
                    grounded.append(ev)
            facts = grounded
            ok_statuses = []
            for st in statuses:
                if evidence_coverage(st.evidence, dialog_text) < self.MIN_EVIDENCE_COVERAGE:
                    rejected.append(self._reject(st, f"цитата «{st.evidence}» не найдена в диалоге"))
                else:
                    ok_statuses.append(st)
            statuses = ok_statuses

        assistant_text = fold(dialog_text or "")

        # «Гастрита и язвы нет» пришло как «хронических болезней нет вообще»: если в цитате статуса
        # раздела названы конкретные пункты справочника, это отрицание этих пунктов, а не всего раздела
        positive_keys = {(n.subject(ev.subject), n.entity(Section(ev.section), ev.name).normalized)
                         for ev in facts if getattr(ev, "name", None) and ev.polarity == "positive"}
        converted_statuses = []
        for st in statuses:
            section = Section(st.section)
            lexicon = n.lexicon_for(section)
            mentions = lexicon.find_mentions(st.evidence) if (lexicon and st.status == "confirmed_absent") else []
            mentions = [m for m in mentions if (n.subject(st.subject), m.normalized) not in positive_keys]
            if not mentions:
                converted_statuses.append(st)
                continue
            for m in mentions:
                parsed, _, _ = parse_extraction({"facts": [{
                    "section": st.section, "subject": st.subject, "name": m.normalized, "polarity": "negative",
                    "evidence": st.evidence}]})
                facts.extend(parsed)
        statuses = converted_statuses

        allergy_reactions = [fold(ev.reaction) for ev in facts
                             if ev.section == "allergies" and getattr(ev, "reaction", None)]

        # Препараты, о которых в этой же реплике говорится как об аллергене (есть аллергия или её нет)
        allergy_keys = {
            (n.subject(ev.subject), n.entity(Section.ALLERGIES, ev.name).normalized)
            for ev in facts if ev.section == "allergies"
        }
        for ev in facts:
            section = Section(ev.section)
            name = getattr(ev, "name", None)
            folded = fold(name) if name else ""

            # «Аллергии на лекарства нет» — это статус раздела, а не аллерген «лекарства»
            if (section == Section.ALLERGIES and ev.polarity == "negative"
                    and (folded.strip(" .") in self.GENERIC_ALLERGENS or clean_name(name) in self.GENERIC_ALLERGENS)):
                statuses.append(SectionStatusEvent(section="allergies", subject=ev.subject,
                                                   status="confirmed_absent", evidence=ev.evidence))
                continue

            if section in self._NOT_DRUG_SECTIONS:
                if folded.startswith(self._ALLERGY_WORDS):
                    rejected.append(self._reject(ev, f"«{name}» — аллергия, она относится к разделу allergies"))
                    continue
                own = n.entity(section, name)
                drug = n.drugs.lookup(name)
                if own.method not in ("exact", "prefix") and drug.method in ("exact", "prefix"):
                    rejected.append(self._reject(
                        ev, f"«{name}» есть в справочнике препаратов ({drug.normalized}) — "
                            f"не относится к разделу {section.value}"))
                    continue
                if section == Section.SYMPTOMS and own.method not in ("exact", "prefix"):
                    diag = n.diagnoses.lookup(name)
                    if diag.method in ("exact", "prefix"):
                        rejected.append(self._reject(
                            ev, f"«{name}» — диагноз ({diag.normalized}), а не симптом"))
                        continue

            # Атрибут образа жизни без слова-маркера: «Лозап бросил» не значит «бросил курить»
            if section == Section.LIFESTYLE:
                marker = self.LIFESTYLE_MARKERS[ev.attribute]
                if not marker.search(fold(ev.evidence)) and not marker.search(assistant_text):
                    rejected.append(self._reject(ev, f"в цитате «{ev.evidence}» нет ничего про {ev.attribute}"))
                    continue

            # Реакция из аллергии — не текущая жалоба: «Ибупрофен плохо переношу — болит желудок»
            if (section == Section.SYMPTOMS and ev.polarity == "positive"
                    and any(fuzz.token_set_ratio(folded, r) >= 80 for r in allergy_reactions)):
                rejected.append(self._reject(ev, f"«{name}» — реакция на препарат из аллергии, а не текущая жалоба"))
                continue

            # «Аллергия на Амоксиклав» — не сведение о том, что человек его не принимает или бросил.
            # Активный приём не трогаем: это прямое утверждение, а не вывод из аллергии.
            if section == Section.MEDICATIONS and (ev.polarity == "negative" or ev.status == "stopped"):
                key = (n.subject(ev.subject), n.entity(Section.MEDICATIONS, name).normalized)
                if key in allergy_keys:
                    rejected.append(self._reject(
                        ev, f"«не принимает {name}» выведено из аллергии, а не сказано покупателем"))
                    continue
            kept.append(ev)
        return kept, statuses, rejected

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
            if (polarity == "negative" and explicit_status in policy.statuses
                    and explicit_status not in policy.active_statuses and explicit_status != "denied"):
                # «Изжога прошла» с polarity=negative и status=resolved: явный неактуальный статус
                # точнее отрицания — было, а теперь нет (а не «никогда не было»)
                polarity = "positive"
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
