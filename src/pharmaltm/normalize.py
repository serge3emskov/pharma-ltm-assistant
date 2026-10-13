"""Нормализация названий по разделам памяти.

LLM передаёт название так, как сказал пользователь («Энап 5 мг»),
а канонический ключ факта («эналаприл») определяет код по справочнику.
Поэтому «Энап» и «эналаприл» в разных диалогах — один и тот же факт.
"""
from __future__ import annotations

from . import reference_data as ref
from .lexicon import Lexicon, NormResult, clean_name, fold, parse_dosage
from .sections import Section


class Normalizer:
    def __init__(
        self,
        drugs: Lexicon | None = None,
        allergens: Lexicon | None = None,
        conditions: Lexicon | None = None,
        symptoms: Lexicon | None = None,
        diagnoses: Lexicon | None = None,
        subjects: dict[str, tuple[str, ...]] | None = None,
    ):
        self.drugs = drugs or Lexicon(ref.DRUGS)
        # Аллергены ищем раньше препаратов: «на пенициллин» — это класс
        self.allergens = allergens or Lexicon([*ref.ALLERGENS, *ref.DRUGS])
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

    def subject(self, raw: str | None) -> str:
        if not raw:
            return "self"
        key = fold(raw)
        if key in self._subjects:
            return self._subjects[key]
        return self._subjects.get(clean_name(raw), key)
