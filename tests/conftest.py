from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pharmaltm import ContextBuilder, MemoryStore, MemoryWriter, Normalizer

USER = "u-test"


class FakeClock:
    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, days: float = 0, hours: float = 0) -> None:
        self.now += timedelta(days=days, hours=hours)


@pytest.fixture(scope="session")
def normalizer() -> Normalizer:
    return Normalizer()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def writer(store, normalizer, clock) -> MemoryWriter:
    return MemoryWriter(store, normalizer, clock)


@pytest.fixture
def builder(store, normalizer, clock) -> ContextBuilder:
    return ContextBuilder(store, normalizer, clock)


def ev(section: str, name: str | None = None, **kw) -> dict:
    """Событие в формате ответа extractor-а."""
    out = {"section": section, "evidence": kw.pop("evidence", "реплика пользователя")}
    if name is not None:
        out["name"] = name
    out.update(kw)
    return out


def absent(section: str, status: str = "confirmed_absent", **kw) -> dict:
    return {"section": section, "status": status, "evidence": kw.pop("evidence", "реплика"), **kw}


def write(writer: MemoryWriter, *facts: dict, statuses: list[dict] | None = None, dialog: str = "d"):
    return writer.write(USER, {"facts": list(facts), "section_statuses": statuses or []}, dialog)
