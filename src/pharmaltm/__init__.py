"""pharmaltm — долгосрочная память аптечного AI-ассистента.

LLM понимает смысл (классифицирует и извлекает), состоянием памяти управляет код.
"""
from .lexicon import LexEntry, Lexicon, NormResult
from .llm import CachedLLM, OpenAICompatLLM, QwenTransformersLLM, ScriptedLLM, extract_json_object
from .memory import AppliedChange, ChangeType, MemoryWriter, WriteReport, classify_transition
from .normalize import Normalizer
from .pipeline import (
    ChatSession,
    Extractor,
    LLMCall,
    PharmacyAssistant,
    QueryClassifier,
    ReadPath,
    SectionClassifier,
    WritePath,
    WriteResult,
)
from .query_rules import classify_query_rules
from .retrieval import ClarifyItem, ContextBuilder, ContextBundle, PurchaseRecord, select_sections, statuses_for
from .schemas import QueryClassification, QueryType, parse_extraction
from .sections import POLICIES, Section
from .store import MemoryStore

__all__ = [
    "AppliedChange", "CachedLLM", "ChangeType", "ChatSession", "ClarifyItem", "ContextBuilder", "ContextBundle",
    "Extractor", "LLMCall", "LexEntry", "Lexicon", "MemoryStore", "MemoryWriter", "NormResult", "Normalizer",
    "OpenAICompatLLM", "POLICIES", "PharmacyAssistant", "PurchaseRecord", "QueryClassification",
    "QueryClassifier", "QueryType", "QwenTransformersLLM", "ReadPath", "ScriptedLLM", "Section",
    "SectionClassifier", "WritePath", "WriteReport", "WriteResult", "classify_query_rules",
    "classify_transition", "extract_json_object", "parse_extraction", "select_sections", "statuses_for",
]

__version__ = "0.2.2"
