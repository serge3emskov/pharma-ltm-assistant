"""Сопоставление свободной формулировки со справочником.

Порядок: точное совпадение → самый длинный префикс по словам →
нечёткое совпадение (опечатки). Нечёткое совпадение принимается,
только если лучший кандидат заметно лучше второго: при неоднозначности
код честно возвращает «не распознано», а не угадывает препарат.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

from rapidfuzz import fuzz, process

_NUM = r"\d+(?:[.,]\d+)?"
DOSAGE_RE = re.compile(
    rf"({_NUM}(?:\s*/\s*{_NUM})*)\s*(мкг|мг|мл|ме|ед|г|%)(?![а-яёa-z])", re.IGNORECASE
)
FORM_WORDS = frozenset(
    """таблетки таблетка табл таб капсулы капсула капс капли сироп спрей мазь гель крем
    раствор порошок шипучие шипучих суспензия свечи суппозитории ампулы драже пастилки
    леденцы мг мкг мл""".split()
)
_NON_WORD = re.compile(r"[^a-zа-я0-9+ ]+")
_SPACES = re.compile(r"\s+")


def fold(text: str) -> str:
    return text.lower().replace("ё", "е").strip()


def parse_dosage(text: str) -> str | None:
    """«Энап 5 мг» → «5 мг»; «Амоксиклав 875/125 мг» → «875/125 мг»."""
    found = []
    for num, unit in DOSAGE_RE.findall(text):
        found.append(_SPACES.sub("", num) + " " + unit.lower())
    return " + ".join(found) if found else None


def clean_name(text: str) -> str:
    """Убирает дозировку, лекарственную форму, числа и пунктуацию."""
    s = fold(text)
    s = DOSAGE_RE.sub(" ", s)
    s = s.replace("р-р", " ").replace("-", " ").replace("/", " ")
    s = _NON_WORD.sub(" ", s)
    tokens = [t for t in s.split() if not t.isdigit() and t not in FORM_WORDS]
    return " ".join(tokens)


@dataclass(frozen=True)
class LexEntry:
    canonical: str
    aliases: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()


Method = Literal["exact", "prefix", "fuzzy", "unmatched"]


@dataclass(frozen=True)
class NormResult:
    raw: str
    cleaned: str
    normalized: str
    method: Method
    score: float = 100.0
    components: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    dosage: str | None = None
    candidates: tuple[str, ...] = field(default_factory=tuple)

    @property
    def matched(self) -> bool:
        return self.method != "unmatched"

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "score": round(self.score, 1),
            "components": list(self.components),
            "classes": list(self.classes),
            "candidates": list(self.candidates),
        }


class Lexicon:
    def __init__(
        self,
        entries: Iterable[LexEntry],
        fuzzy_cutoff: float = 85.0,
        min_fuzzy_len: int = 5,
        ambiguity_margin: float = 3.0,
    ):
        self.fuzzy_cutoff = fuzzy_cutoff
        self.min_fuzzy_len = min_fuzzy_len
        self.ambiguity_margin = ambiguity_margin
        self._index: dict[str, LexEntry] = {}
        for entry in entries:
            for alias in (entry.canonical, *entry.aliases):
                key = clean_name(alias)
                if key and key not in self._index:  # первая запись в списке имеет приоритет
                    self._index[key] = entry
        self._keys = list(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def _result(self, raw: str, cleaned: str, entry: LexEntry, method: Method, score: float) -> NormResult:
        components = entry.components or (entry.canonical,)
        return NormResult(
            raw=raw, cleaned=cleaned, normalized=entry.canonical, method=method, score=score,
            components=components, classes=entry.classes, dosage=parse_dosage(raw),
        )

    def _fuzzy(self, query: str) -> tuple[LexEntry | None, float, tuple[str, ...]]:
        if len(query) < self.min_fuzzy_len:
            return None, 0.0, ()
        hits = process.extract(query, self._keys, scorer=fuzz.ratio, limit=8, score_cutoff=self.fuzzy_cutoff)
        best: dict[str, tuple[float, LexEntry]] = {}
        for key, score, _ in hits:
            entry = self._index[key]
            if entry.canonical not in best or score > best[entry.canonical][0]:
                best[entry.canonical] = (score, entry)
        ranked = sorted(best.values(), key=lambda x: -x[0])
        if not ranked:
            return None, 0.0, ()
        names = tuple(e.canonical for _, e in ranked)
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.ambiguity_margin:
            return None, ranked[0][0], names  # неоднозначно — не угадываем
        return ranked[0][1], ranked[0][0], names

    def lookup(self, raw: str) -> NormResult:
        cleaned = clean_name(raw)
        if not cleaned:
            return NormResult(raw=raw, cleaned="", normalized=fold(raw), method="unmatched", score=0.0,
                              dosage=parse_dosage(raw))
        if cleaned in self._index:
            return self._result(raw, cleaned, self._index[cleaned], "exact", 100.0)

        tokens = cleaned.split()
        for n in range(len(tokens) - 1, 0, -1):
            prefix = " ".join(tokens[:n])
            if prefix in self._index:
                return self._result(raw, cleaned, self._index[prefix], "prefix", 100.0)

        candidates: tuple[str, ...] = ()
        for query in dict.fromkeys((cleaned, tokens[0])):
            entry, score, names = self._fuzzy(query)
            candidates = candidates or names
            if entry is not None:
                return replace(self._result(raw, cleaned, entry, "fuzzy", score), candidates=names)

        return NormResult(raw=raw, cleaned=cleaned, normalized=cleaned, method="unmatched", score=0.0,
                          dosage=parse_dosage(raw), candidates=candidates)
