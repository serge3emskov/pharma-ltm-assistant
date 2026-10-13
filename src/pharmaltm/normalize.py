"""Нормализация названий по разделам памяти.

LLM передаёт название так, как сказал пользователь («Энап 5 мг»),
а канонический ключ факта («эналаприл») определяет код по справочнику.
Поэтому «Энап» и «эналаприл» в разных диалогах — один и тот же факт.
"""
from __future__ import annotations

import re

from . import reference_data as ref
from .catalog import DrugCatalog
from .lexicon import Lexicon, NormResult, clean_name, fold, parse_dosage
from .sections import Section


class Normalizer:
    """Справочники всех разделов. Препараты — из DrugCatalog (стартовый набор или + ГРЛС)."""

    def __init__(
        self,
        drugs: Lexicon | None = None,
        allergens: Lexicon | None = None,
        conditions: Lexicon | None = None,
        symptoms: Lexicon | None = None,
        diagnoses: Lexicon | None = None,
        subjects: dict[str, tuple[str, ...]] | None = None,
        catalog: DrugCatalog | None = None,
    ):
        self.catalog = catalog or DrugCatalog()
        self.drugs = drugs or self.catalog.lexicon()
        # Аллергены ищем раньше препаратов: «на пенициллин» — это класс
        self.allergens = allergens or Lexicon(
            [*ref.ALLERGENS, *self.catalog.entries()],
            mention_keys={clean_name(a) for e in ref.ALLERGENS for a in (e.canonical, *e.aliases)}
            | self.catalog.mention_keys(self.catalog.non_drug_words()),
        )
        self.conditions = conditions or Lexicon(ref.CONDITIONS)
        self.symptoms = symptoms or Lexicon(ref.SYMPTOMS)
        self.diagnoses = diagnoses or Lexicon(ref.DIAGNOSES)
        self._subjects: dict[str, str] = {}
        for canonical, aliases in (subjects or ref.SUBJECTS).items():
            for alias in (canonical, *aliases):
                self._subjects[fold(alias)] = canonical

    def lexicon_for(self, section: Section) -> Lexicon | None:
        return {
            Section.MEDICATIONS: self.drugs,
            Section.ALLERGIES: self.allergens,
            Section.CHRONIC_CONDITIONS: self.conditions,
            Section.SYMPTOMS: self.symptoms,
            Section.REPORTED_DIAGNOSES: self.diagnoses,
        }.get(section)

    def entity(self, section: Section, name: str) -> NormResult:
        lexicon = self.lexicon_for(section)
        if lexicon is None:  # special_states: канон уже обеспечен схемой
            key = fold(name)
            return NormResult(raw=name, cleaned=key, normalized=name, method="exact", dosage=parse_dosage(name))
        return lexicon.lookup(name)

    @classmethod
    def from_catalog(cls, catalog: DrugCatalog) -> "Normalizer":
        return cls(catalog=catalog)

    def drug_mentions(self, text: str) -> list[NormResult]:
        """Препараты из справочника, упомянутые в тексте: для блока СПРАВОЧНИК в контексте ответа."""
        return self.drugs.find_mentions(text)

    _CHILD_RE = re.compile(r"малыш|младен|грудничок|новорожд|годовал|ребен|ребят|детк")

    def subject(self, raw: str | None) -> str:
        if not raw:
            return "self"
        key = fold(raw)
        if key in self._subjects:
            return self._subjects[key]
        cleaned = clean_name(raw)
        if cleaned in self._subjects:
            return self._subjects[cleaned]
        if self._CHILD_RE.search(key):  # «годовалый малыш», «наш младенец»
            return "ребёнок"
        return cleaned or key
