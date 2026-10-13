"""Хранилище LTM на SQLite.

Текущее состояние факта (facts) и история событий (events) лежат отдельно:
для ответа обычно хватает состояния, а историю можно восстановить по запросу.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Iterator

from .sections import Section

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id           TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    subject           TEXT NOT NULL,
    section           TEXT NOT NULL,
    key               TEXT NOT NULL,
    status            TEXT NOT NULL,
    data_json         TEXT NOT NULL,
    raw_names_json    TEXT NOT NULL,
    norm_json         TEXT NOT NULL,
    first_seen_at     TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    last_confirmed_at TEXT NOT NULL,
    mentions          INTEGER NOT NULL,
    UNIQUE (user_id, subject, section, key)
);
CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT NOT NULL UNIQUE,
    user_id      TEXT NOT NULL,
    subject      TEXT NOT NULL,
    section      TEXT NOT NULL,
    fact_id      TEXT,
    change_type  TEXT NOT NULL,
    dialog_id    TEXT,
    created_at   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    before_json  TEXT,
    after_json   TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_fact ON events (fact_id, seq);
CREATE INDEX IF NOT EXISTS ix_events_user ON events (user_id, seq);
CREATE TABLE IF NOT EXISTS section_states (
    user_id             TEXT NOT NULL,
    subject             TEXT NOT NULL,
    section             TEXT NOT NULL,
    state               TEXT,
    needs_clarification INTEGER NOT NULL DEFAULT 0,
    updated_at          TEXT NOT NULL,
    PRIMARY KEY (user_id, subject, section)
);
CREATE TABLE IF NOT EXISTS rejections (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    dialog_id  TEXT,
    created_at TEXT NOT NULL,
    kind       TEXT NOT NULL,
    raw_json   TEXT NOT NULL,
    errors     TEXT NOT NULL
);
"""


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def _ts(dt: datetime) -> str:
    return dt.isoformat()


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Fact:
    fact_id: str
    user_id: str
    subject: str
    section: Section
    key: str
    status: str
    data: dict[str, Any]
    raw_names: list[str]
    norm: dict[str, Any]
    first_seen_at: datetime
    updated_at: datetime
    last_confirmed_at: datetime
    mentions: int = 1

    def snapshot(self) -> dict[str, Any]:
        """Состояние факта для истории: что было до и после события."""
        return {"status": self.status, "data": dict(self.data)}


@dataclass
class EventRecord:
    seq: int
    event_id: str
    user_id: str
    subject: str
    section: Section
    fact_id: str | None
    change_type: str
    dialog_id: str | None
    created_at: datetime
    payload: dict[str, Any]
    before: dict[str, Any] | None
    after: dict[str, Any] | None


@dataclass
class SectionState:
    user_id: str
    subject: str
    section: Section
    state: str | None  # None | "confirmed_absent" | "dont_know"
    needs_clarification: bool = False
    updated_at: datetime | None = None


@dataclass
class RejectionRecord:
    seq: int
    user_id: str
    dialog_id: str | None
    created_at: datetime
    kind: str
    raw: Any
    errors: list[str] = field(default_factory=list)


class MemoryStore:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._depth = 0

    # ------------------------------------------------------------------ транзакции
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Вложенные вызовы допустимы: коммит делает внешний уровень."""
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.rollback()
            raise
        else:
            self._depth -= 1
            if self._depth == 0:
                self.conn.commit()

    # ------------------------------------------------------------------ факты
    @staticmethod
    def _fact(row: sqlite3.Row) -> Fact:
        return Fact(
            fact_id=row["fact_id"], user_id=row["user_id"], subject=row["subject"],
            section=Section(row["section"]), key=row["key"], status=row["status"],
            data=json.loads(row["data_json"]), raw_names=json.loads(row["raw_names_json"]),
            norm=json.loads(row["norm_json"]), first_seen_at=_dt(row["first_seen_at"]),
            updated_at=_dt(row["updated_at"]), last_confirmed_at=_dt(row["last_confirmed_at"]),
            mentions=row["mentions"],
        )

    def get_fact(self, user_id: str, subject: str, section: Section, key: str) -> Fact | None:
        row = self.conn.execute(
            "SELECT * FROM facts WHERE user_id=? AND subject=? AND section=? AND key=?",
            (user_id, subject, section.value, key),
        ).fetchone()
        return self._fact(row) if row else None

    def get_fact_by_id(self, fact_id: str) -> Fact | None:
        row = self.conn.execute("SELECT * FROM facts WHERE fact_id=?", (fact_id,)).fetchone()
        return self._fact(row) if row else None

    def save_fact(self, fact: Fact) -> None:
        self.conn.execute(
            """INSERT INTO facts (fact_id, user_id, subject, section, key, status, data_json,
                   raw_names_json, norm_json, first_seen_at, updated_at, last_confirmed_at, mentions)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(fact_id) DO UPDATE SET
                   status=excluded.status, data_json=excluded.data_json,
                   raw_names_json=excluded.raw_names_json, norm_json=excluded.norm_json,
                   updated_at=excluded.updated_at, last_confirmed_at=excluded.last_confirmed_at,
                   mentions=excluded.mentions""",
            (
                fact.fact_id, fact.user_id, fact.subject, fact.section.value, fact.key, fact.status,
                _dumps(fact.data), _dumps(fact.raw_names), _dumps(fact.norm),
                _ts(fact.first_seen_at), _ts(fact.updated_at), _ts(fact.last_confirmed_at), fact.mentions,
            ),
        )

    def list_facts(
        self,
        user_id: str,
        subject: str | None = None,
        sections: Iterable[Section] | None = None,
        statuses: Iterable[str] | None = None,
    ) -> list[Fact]:
        sql = "SELECT * FROM facts WHERE user_id=?"
        params: list[Any] = [user_id]
        if subject is not None:
            sql += " AND subject=?"
            params.append(subject)
        if sections is not None:
            secs = [s.value for s in sections]
            if not secs:
                return []
            sql += f" AND section IN ({','.join('?' * len(secs))})"
            params += secs
        if statuses is not None:
            sts = list(statuses)
            if not sts:
                return []
            sql += f" AND status IN ({','.join('?' * len(sts))})"
            params += sts
        sql += " ORDER BY subject, section, first_seen_at, key"
        return [self._fact(r) for r in self.conn.execute(sql, params)]

    def subjects(self, user_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT subject FROM facts WHERE user_id=? UNION SELECT subject FROM section_states WHERE user_id=?",
            (user_id, user_id),
        )
        return sorted(r[0] for r in rows)

    # ------------------------------------------------------------------ события
    def add_event(
        self,
        *,
        user_id: str,
        subject: str,
        section: Section,
        fact_id: str | None,
        change_type: str,
        dialog_id: str | None,
        created_at: datetime,
        payload: dict[str, Any],
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> str:
        event_id = new_id()
        self.conn.execute(
            """INSERT INTO events (event_id, user_id, subject, section, fact_id, change_type, dialog_id,
                   created_at, payload_json, before_json, after_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id, user_id, subject, section.value, fact_id, change_type, dialog_id, _ts(created_at),
                _dumps(payload), None if before is None else _dumps(before),
                None if after is None else _dumps(after),
            ),
        )
        return event_id

    @staticmethod
    def _event(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            seq=row["seq"], event_id=row["event_id"], user_id=row["user_id"], subject=row["subject"],
            section=Section(row["section"]), fact_id=row["fact_id"], change_type=row["change_type"],
            dialog_id=row["dialog_id"], created_at=_dt(row["created_at"]),
            payload=json.loads(row["payload_json"]),
            before=json.loads(row["before_json"]) if row["before_json"] else None,
            after=json.loads(row["after_json"]) if row["after_json"] else None,
        )

    def history(self, fact_id: str) -> list[EventRecord]:
        rows = self.conn.execute("SELECT * FROM events WHERE fact_id=? ORDER BY seq", (fact_id,))
        return [self._event(r) for r in rows]

    def events(self, user_id: str, dialog_id: str | None = None) -> list[EventRecord]:
        sql, params = "SELECT * FROM events WHERE user_id=?", [user_id]
        if dialog_id is not None:
            sql += " AND dialog_id=?"
            params.append(dialog_id)
        return [self._event(r) for r in self.conn.execute(sql + " ORDER BY seq", params)]

    # ------------------------------------------------------------------ статусы разделов
    def get_section_state(self, user_id: str, subject: str, section: Section) -> SectionState:
        row = self.conn.execute(
            "SELECT * FROM section_states WHERE user_id=? AND subject=? AND section=?",
            (user_id, subject, section.value),
        ).fetchone()
        if not row:
            return SectionState(user_id=user_id, subject=subject, section=section, state=None)
        return SectionState(
            user_id=user_id, subject=subject, section=section, state=row["state"],
            needs_clarification=bool(row["needs_clarification"]), updated_at=_dt(row["updated_at"]),
        )

    def list_section_states(self, user_id: str) -> list[SectionState]:
        rows = self.conn.execute(
            "SELECT * FROM section_states WHERE user_id=? ORDER BY subject, section", (user_id,)
        )
        return [
            SectionState(user_id=user_id, subject=r["subject"], section=Section(r["section"]), state=r["state"],
                         needs_clarification=bool(r["needs_clarification"]), updated_at=_dt(r["updated_at"]))
            for r in rows
        ]

    def save_section_state(self, st: SectionState, now: datetime) -> None:
        if st.state is None and not st.needs_clarification:
            self.conn.execute(
                "DELETE FROM section_states WHERE user_id=? AND subject=? AND section=?",
                (st.user_id, st.subject, st.section.value),
            )
            return
        self.conn.execute(
            """INSERT INTO section_states (user_id, subject, section, state, needs_clarification, updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(user_id, subject, section) DO UPDATE SET
                   state=excluded.state, needs_clarification=excluded.needs_clarification,
                   updated_at=excluded.updated_at""",
            (st.user_id, st.subject, st.section.value, st.state, int(st.needs_clarification), _ts(now)),
        )

    # ------------------------------------------------------------------ отклонённые события
    def add_rejection(self, user_id: str, dialog_id: str | None, created_at: datetime,
                      kind: str, raw: Any, errors: list[str]) -> None:
        self.conn.execute(
            "INSERT INTO rejections (user_id, dialog_id, created_at, kind, raw_json, errors) VALUES (?,?,?,?,?,?)",
            (user_id, dialog_id, _ts(created_at), kind, _dumps(raw), _dumps(errors)),
        )

    def rejections(self, user_id: str) -> list[RejectionRecord]:
        rows = self.conn.execute("SELECT * FROM rejections WHERE user_id=? ORDER BY seq", (user_id,))
        return [
            RejectionRecord(seq=r["seq"], user_id=r["user_id"], dialog_id=r["dialog_id"],
                            created_at=_dt(r["created_at"]), kind=r["kind"],
                            raw=json.loads(r["raw_json"]), errors=json.loads(r["errors"]))
            for r in rows
        ]
