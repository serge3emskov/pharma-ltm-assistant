"""pharmaltm — долгосрочная память аптечного AI-ассистента.

LLM понимает смысл (классифицирует и извлекает), состоянием памяти управляет код.
"""
from .lexicon import LexEntry, Lexicon, NormResult
from .memory import AppliedChange, ChangeType, MemoryWriter, WriteReport, classify_transition
from .normalize import Normalizer
from .query_rules import classify_query_rules
from .retrieval import ClarifyItem, ContextBuilder, ContextBundle, PurchaseRecord, select_sections, statuses_for
from .schemas import QueryClassification, QueryType, parse_extraction
from .sections import POLICIES, Section
from .store import MemoryStore

__all__ = [
    "AppliedChange", "ChangeType", "ClarifyItem", "ContextBuilder", "ContextBundle", "LexEntry", "Lexicon",
    "MemoryStore", "MemoryWriter", "NormResult", "Normalizer", "POLICIES", "PurchaseRecord",
    "QueryClassification", "QueryType", "Section", "WriteReport", "classify_query_rules",
    "classify_transition", "parse_extraction", "select_sections", "statuses_for",
]

__version__ = "0.1.0"
