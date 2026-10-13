"""Быстрая проверка LLM-шагов на dev-наборе.

Это не полноценная оценка (она — на этапе 3), а дымовой тест промптов:
16 диалогов и 20 запросов с эталонами. Сравнивается то, что важно
для памяти: итоговое состояние фактов после write-path, а не дословный JSON.

GoldLLM отвечает эталонами из dev-набора. С ним метрики должны быть 100%:
так проверяется согласованность самих эталонов, кода и нормализации.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .llm import Messages
from .memory import MemoryWriter
from .normalize import Normalizer
from .pipeline import QueryClassifier, WritePath, WriteResult
from .prompts import format_dialog
from .query_rules import classify_query_rules
from .store import MemoryStore

DEV_USER = "dev-user"


def data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_dev(directory: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    d = Path(directory) if directory else data_dir()
    return load_jsonl(d / "dev_dialogs.jsonl"), load_jsonl(d / "dev_queries.jsonl")


# --------------------------------------------------------------------------- эталонный бэкенд


class GoldLLM:
    """Отвечает эталонами dev-набора. Неизвестный диалог — пустой ответ, неизвестный запрос — не-JSON."""

    name = "gold"

    def __init__(self, dialogs: list[dict], queries: list[dict],
                 answer_text: str = "[эталонный бэкенд: здесь будет ответ модели]"):
        self.dialogs = {format_dialog(d["turns"]): d for d in dialogs}
        self.queries = {q["text"]: q for q in queries}
        self.answer_text = answer_text

    def chat(self, messages: Messages, *, max_new_tokens: int = 512, json_mode: bool = False,
             meta: dict[str, Any] | None = None) -> str:
        meta = meta or {}
        task = meta.get("task")
        last = messages[-1]["content"]
        if task == "classify_sections":
            d = self.dialogs.get(last.removeprefix("Диалог:\n"))
            return json.dumps({"sections": d["gold_sections"] if d else []}, ensure_ascii=False)
        if task == "extract":
            d = self.dialogs.get(last.removeprefix("Диалог:\n"))
            gold = d["gold_extraction"] if d else {}
            sec = meta.get("section")
            out = {k: [x for x in gold.get(k, []) if x["section"] == sec] for k in ("facts", "section_statuses")}
            return json.dumps(out, ensure_ascii=False)
        if task == "classify_query":
            text = last.rsplit("Сообщение покупателя: ", 1)[-1]
            q = self.queries.get(text)
            if not q:
                return "не знаю"
            return json.dumps({"query_types": q["query_types"], "subject": q["subject"]}, ensure_ascii=False)
        return self.answer_text


# --------------------------------------------------------------------------- write-path


@dataclass
class DialogResult:
    id: str
    note: str
    gold_sections: set[str]
    pred_sections: set[str]
    missing_facts: set[tuple]
    extra_facts: set[tuple]
    missing_states: set[tuple]
    extra_states: set[tuple]
    data_errors: list[str]
    write: WriteResult

    @property
    def ok(self) -> bool:
        return not (self.missing_facts or self.extra_facts or self.missing_states or self.extra_states
                    or self.data_errors)

    def row(self) -> dict[str, Any]:
        s = self.write.summary()
        return {
            "id": self.id,
            "ok": self.ok,
            "note": self.note,
            "sections_gold": sorted(self.gold_sections),
            "sections_pred": sorted(self.pred_sections),
            "missing": sorted(self.missing_facts | self.missing_states),
            "extra": sorted(self.extra_facts | self.extra_states),
            "data_errors": self.data_errors,
            "rejected": s["rejected"],
            "off_section": s["off_section"],
            "retries": s["retries"],
            "latency_s": s["latency_s"],
        }


def _final_state(store: MemoryStore, user: str) -> tuple[set[tuple], set[tuple]]:
    facts = {(f.section.value, f.subject, f.key, f.status) for f in store.list_facts(user)}
    states = {(s.section.value, s.subject, s.state) for s in store.list_section_states(user) if s.state}
    return facts, states


def run_write_dev(
    llm: Any,
    dialogs: Iterable[dict],
    normalizer: Normalizer | None = None,
    clock: Callable[[], datetime] | None = None,
) -> list[DialogResult]:
    normalizer = normalizer or Normalizer()
    clock = clock or (lambda: datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc))
    results = []
    for d in dialogs:
        store = MemoryStore()
        writer = MemoryWriter(store, normalizer, clock)
        if d.get("setup"):
            writer.write(DEV_USER, d["setup"], "setup")
        write = WritePath(llm, writer).process(DEV_USER, d["turns"], d["id"])
        facts, states = _final_state(store, DEV_USER)
        exp_facts = {tuple(x) for x in d["expected_facts"]}
        exp_states = {tuple(x) for x in d["expected_states"]}
        data_errors = []
        for section, subject, key, fld, value in d.get("expected_data", []):
            fact = next((f for f in store.list_facts(DEV_USER, subject) if f.section.value == section and f.key == key),
                        None)
            got = None if fact is None else fact.data.get(fld)
            if str(got) != str(value):
                data_errors.append(f"{section}/{key}.{fld}: ожидалось {value!r}, получено {got!r}")
        results.append(DialogResult(
            id=d["id"], note=d.get("note", ""),
            gold_sections=set(d["gold_sections"]), pred_sections={s.value for s in write.sections},
            missing_facts=exp_facts - facts, extra_facts=facts - exp_facts,
            missing_states=exp_states - states, extra_states=states - exp_states,
            data_errors=data_errors, write=write,
        ))
    return results


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def summarize_write(results: list[DialogResult], dialogs: list[dict]) -> dict[str, Any]:
    by_id = {d["id"]: d for d in dialogs}
    s_tp = sum(len(r.gold_sections & r.pred_sections) for r in results)
    s_fp = sum(len(r.pred_sections - r.gold_sections) for r in results)
    s_fn = sum(len(r.gold_sections - r.pred_sections) for r in results)
    f_fn = sum(len(r.missing_facts) for r in results)
    f_fp = sum(len(r.extra_facts) for r in results)
    f_tp = sum(len(by_id[r.id]["expected_facts"]) for r in results) - f_fn
    st_fn = sum(len(r.missing_states) for r in results)
    st_fp = sum(len(r.extra_states) for r in results)
    st_tp = sum(len(by_id[r.id]["expected_states"]) for r in results) - st_fn
    summaries = [r.write.summary() for r in results]
    return {
        "dialogs": len(results),
        "dialogs_fully_correct": sum(r.ok for r in results),
        "sections": _prf(s_tp, s_fp, s_fn),
        "facts": _prf(f_tp, f_fp, f_fn),
        "section_states": _prf(st_tp, st_fp, st_fn),
        "data_errors": sum(len(r.data_errors) for r in results),
        "rejected_events": sum(s["rejected"] for s in summaries),
        "off_section_dropped": sum(s["off_section"] for s in summaries),
        "llm_calls": sum(s["llm_calls"] for s in summaries),
        "retries": sum(s["retries"] for s in summaries),
        "failed_calls": sum(1 for r in results for c in r.write.calls if not c.ok),
        "latency_s_total": round(sum(s["latency_s"] for s in summaries), 1),
    }


# --------------------------------------------------------------------------- классификатор запроса


@dataclass
class QueryResult:
    id: str
    text: str
    gold_types: set[str]
    pred_types: set[str]
    gold_subject: str
    pred_subject: str
    fallback_used: bool
    raw: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def types_ok(self) -> bool:
        return self.gold_types == self.pred_types

    @property
    def subject_ok(self) -> bool:
        return self.gold_subject == self.pred_subject

    def row(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "types_ok": self.types_ok, "subject_ok": self.subject_ok,
                "gold": sorted(self.gold_types), "pred": sorted(self.pred_types),
                "subject_gold": self.gold_subject, "subject_pred": self.pred_subject,
                "fallback": self.fallback_used}


def run_query_dev(classifier: QueryClassifier | None, queries: Iterable[dict],
                  normalizer: Normalizer | None = None) -> list[QueryResult]:
    """classifier=None — baseline на правилах."""
    normalizer = normalizer or Normalizer()
    out = []
    for q in queries:
        if classifier is None:
            qc, call, fallback = classify_query_rules(q["text"]), None, True
        else:
            qc, call, fallback = classifier(q["text"])
        out.append(QueryResult(
            id=q["id"], text=q["text"], gold_types=set(q["query_types"]),
            pred_types={t.value for t in qc.query_types}, gold_subject=normalizer.subject(q["subject"]),
            pred_subject=normalizer.subject(qc.subject), fallback_used=fallback and classifier is not None,
            raw=call.raw if call else None,
        ))
    return out


def summarize_queries(results: list[QueryResult]) -> dict[str, Any]:
    tp = sum(len(r.gold_types & r.pred_types) for r in results)
    fp = sum(len(r.pred_types - r.gold_types) for r in results)
    fn = sum(len(r.gold_types - r.pred_types) for r in results)
    n = len(results) or 1
    return {
        "queries": len(results),
        "types_exact_match": round(sum(r.types_ok for r in results) / n, 3),
        "types": _prf(tp, fp, fn),
        "subject_accuracy": round(sum(r.subject_ok for r in results) / n, 3),
        "fallbacks": sum(r.fallback_used for r in results),
    }
