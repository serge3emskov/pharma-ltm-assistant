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
_TOKEN = re.compile(r"[a-zа-я0-9]+")


def fold(text: str) -> str:
    return text.lower().replace("ё", "е").strip()


def evidence_coverage(evidence: str, text: str, min_token_len: int = 3, token_match: float = 85.0) -> float:
    """Доля слов цитаты, найденных в тексте диалога (с допуском на словоформу).

    Сравнение по словам, а не подстрокой: модель может сократить цитату
    («пьёт Диротон» из «пьёт Сиофор и Диротон») — это не выдумка. А «не кормлю»
    в диалоге, где о кормлении нет ни слова, — выдумка.
    """
    words = [w for w in _TOKEN.findall(fold(evidence)) if len(w) >= min_token_len]
    if not words:
        return 1.0
    vocab = set(_TOKEN.findall(fold(text)))
    found = 0
    for w in words:
        if w in vocab or any(abs(len(v) - len(w)) <= 3 and fuzz.ratio(w, v) >= token_match for v in vocab):
            found += 1
    return found / len(words)


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

    # Окончания, которые отбрасываем при поиске упоминаний: «Конкором» → «конкор», «Лозапа» → «лозап»
    _ENDINGS = ("ами", "ями", "ом", "ем", "ой", "ей", "ов", "ам", "ах", "а", "у", "е", "ы", "и", "ю")

    def find_mentions(self, text: str, min_len: int = 4) -> list[NormResult]:
        """Упоминания записей справочника в свободном тексте (только точные совпадения и падежные окончания).

        Нечёткий поиск здесь не используется: в обычной речи слишком много похожих слов.
        Записи короче min_len пропускаются («миг» — это ещё и обычное слово).
        """
        words = _NON_WORD.sub(" ", fold(text).replace("-", " ")).split()
        found: dict[str, NormResult] = {}
        i = 0
        while i < len(words):
            hit, step = None, 1
            for n in (3, 2, 1):
                chunk = words[i:i + n]
                if len(chunk) < n:
                    continue
                last = chunk[-1]
                stems = [last[: -len(e)] for e in self._ENDINGS if last.endswith(e)]
                # «Но-шпу» → «но шпа»: женский род в косвенном падеже
                variants = [last, *stems, *(st + "а" for st in stems), *(st + "я" for st in stems)]
                for v in variants:
                    gram = " ".join(chunk[:-1] + [v])
                    if v and len(gram) >= min_len and gram in self._index:
                        hit, step = gram, n
                        break
                if hit:
                    break
            if hit is not None:
                entry = self._index[hit]
                if entry.canonical not in found:
                    raw = " ".join(words[i:i + step])
                    found[entry.canonical] = self._result(raw, hit, entry, "exact", 100.0)
            i += step
        return list(found.values())
