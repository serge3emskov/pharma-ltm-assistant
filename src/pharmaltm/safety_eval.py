"""Этап 4: оценка проверки по справочнику.

Три независимых измерения:

1. **Правила** (без LLM): проверочные случаи data/safety/cases.jsonl — память покупателя + вопрос →
   полный список ожидаемых предупреждений. Считаются точность и полнота: лишнее предупреждение
   — такая же ошибка, как пропущенное.
2. **Повтор сохранённых ответов** (без GPU): ответы моделей из отчёта этапа 3 проходят проверку
   после ответа — видно, что код дописал бы к ответам, которые модель уже дала.
3. **E2E с моделью** (GPU): случаи с `answer_checks` — ответ модели без блока проверки в контексте
   (как на этапе 3) и с ним; для обоих — что модель учла сама и что дописал код.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .followup import clarify_check
from .evaluation import GoldenSet, check_answer, gold_backend, run_golden
from .llm import chat_many
from .memory import MemoryWriter
from .normalize import Normalizer
from .prompts import answer_messages
from .retrieval import ContextBuilder
from .safety import LEVEL_RANK, SafetyChecker
from .schemas import QueryType
from .store import MemoryStore

USER = "safety-user"
NOW = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)


def cases_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "safety" / "cases.jsonl"


def load_cases(path: str | Path | None = None) -> list[dict]:
    with Path(path or cases_path()).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def case_store(case: dict, normalizer: Normalizer) -> MemoryStore:
    """Память покупателя из разметки случая — тем же MemoryWriter, что и в пайплайне."""
    store = MemoryStore()
    writer = MemoryWriter(store, normalizer, lambda: NOW)
    report = writer.write(USER, case["setup"], f"{case['id']}#setup")
    if report.rejected:
        raise ValueError(f"{case['id']}: разметка памяти не прошла проверку: {report.rejected[0].errors}")
    return store


# =========================================================================== 1. правила


def run_cases(checker: SafetyChecker, cases: list[dict], normalizer: Normalizer | None = None) -> list[dict]:
    normalizer = normalizer or Normalizer(catalog=checker.catalog)
    rows = []
    for c in cases:
        store = case_store(c, normalizer)
        subject = normalizer.subject(c["subject"])
        profile = checker.profile(store, USER, subject, NOW)
        report = checker.report(profile, c["question"])
        # «Только по рецепту» зависит от справочника (известно лишь с ГРЛС), а не от правил — считаем отдельно
        rx = [a for a in report.alerts if a.trigger_kind == "dispensing"]
        got = {(a.rule_id, a.drug.canonical) for a in report.alerts if a.trigger_kind != "dispensing"}
        exp = {tuple(x) for x in c["expect"]}
        rows.append({
            "id": c["id"], "title": c["title"], "ok": got == exp,
            "missing": sorted(map(list, exp - got)), "extra": sorted(map(list, got - exp)),
            "tp": len(got & exp), "fp": len(got - exp), "fn": len(exp - got),
            "question_drugs": [d.label for d in report.question_drugs],
            "rx_only": [a.drug.label for a in rx],
            "alerts": [a.render() for a in report.alerts],
            "restrictions": [r.render() for r in report.restrictions],
        })
    return rows


def summarize_cases(rows: list[dict]) -> dict[str, Any]:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return {
        "cases": len(rows),
        "exact": sum(r["ok"] for r in rows),
        "negative_controls": sum(1 for r in rows if r["tp"] + r["fn"] == 0),
        "alerts_expected": tp + fn,
        "precision": round(tp / (tp + fp), 3) if tp + fp else 1.0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 1.0,
        "failed": [r["id"] for r in rows if not r["ok"]],
    }


# =========================================================================== 2. повтор ответов этапа 3


def replay_answers(stage3_report: dict, golden: GoldenSet, checker: SafetyChecker,
                   normalizer: Normalizer | None = None) -> list[dict]:
    """Ответы из отчёта этапа 3 → проверка после ответа.

    Память восстанавливается эталонной разметкой golden set (тем, что должно было быть записано),
    поэтому повтор показывает работу проверки, а не ошибки памяти конкретной модели.
    """
    normalizer = normalizer or Normalizer(catalog=checker.catalog)
    run = run_golden(gold_backend(golden), golden, normalizer=normalizer, run_answers=False, safety=checker)
    probes = {(p.scenario_id, p.step): p for r in run.scenarios for p in r.probes}
    reports = stage3_report.get("reports") or {stage3_report.get("model", "model"): stage3_report}
    rows = []
    for model, rep in reports.items():
        for pr in rep.get("probes", []):
            if not pr.get("answer"):
                continue
            p = probes.get((pr["scenario"], pr["step"]))
            if p is None or p.bundle_llm.safety is None:
                continue
            safety = p.bundle_llm.safety
            post = checker.post_check(pr["answer"], safety.profile, safety.question_drugs)
            rows.append({
                "model": model, "scenario": pr["scenario"], "question": pr["text"], "answer": pr["answer"],
                "profile": safety.profile.as_dict(),
                "answer_drugs": [d.label for d in post.answer_drugs],
                "appended": [a.render() for a in post.appended],
                "covered": [a.render() for a in post.covered],
                "declined": [a.render() for a in post.declined],
                "doses": post.doses,
                "final": post.text,
            })
    return rows


# =========================================================================== 3. E2E с моделью


def build_e2e_prompts(checker: SafetyChecker, cases: list[dict], normalizer: Normalizer,
                      with_safety: bool) -> list[dict]:
    items = []
    for c in cases:
        if not c["answer_checks"]:
            continue
        store = case_store(c, normalizer)
        subject = normalizer.subject(c["subject"])
        builder = ContextBuilder(store, normalizer, lambda: NOW, safety=checker if with_safety else None)
        bundle = builder.build(USER, [QueryType(t) for t in c["query_types"]], subject, reference_text=c["question"])
        # Профиль и препараты из вопроса нужны и без блока в контексте: проверка после ответа работает всегда
        report = checker.report(checker.profile(store, USER, subject, NOW), c["question"])
        items.append({"case": c, "context": bundle.render(), "report": report, "bundle": bundle,
                      "prompt": answer_messages(bundle.render(), [], c["question"])})
    return items


def run_safety_e2e(llm: Any, checker: SafetyChecker, cases: list[dict], *, normalizer: Normalizer | None = None,
                   conditions: tuple[str, ...] = ("baseline", "safety"), batch_size: int = 8,
                   max_new_tokens: int = 512) -> list[dict]:
    """baseline — контекст как на этапе 3; safety — с блоком ПРОВЕРКА ПО СПРАВОЧНИКУ.

    В обоих условиях к ответу применяется проверка после ответа: так видно, сколько предупреждений
    модель учла сама, а сколько пришлось дописать коду.
    """
    normalizer = normalizer or Normalizer(catalog=checker.catalog)
    rows = []
    for cond in conditions:
        items = build_e2e_prompts(checker, cases, normalizer, with_safety=(cond == "safety"))
        answers = chat_many(llm, [it["prompt"] for it in items], max_new_tokens=max_new_tokens, json_mode=False,
                            metas=[{"task": "answer"}] * len(items), batch_size=batch_size)
        for it, a in zip(items, answers):
            c, report = it["case"], it["report"]
            raw = a.strip()
            post = checker.post_check(raw, report.profile, report.question_drugs)
            clar = clarify_check(post.text, it["bundle"].clarify, it["bundle"].subject, bool(post.answer_drugs))
            rows.append({
                "id": c["id"], "title": c["title"], "condition": cond, "question": c["question"],
                "context": it["context"], "raw": raw, "final": clar.text,
                "checks_raw": check_answer(raw, c["answer_checks"]),
                "checks_final": check_answer(clar.text, c["answer_checks"]),
                "clarify": clar.as_dict(),
                "expected_alerts": [a.render() for a in report.alerts],
                "answer_drugs": [d.label for d in post.answer_drugs],
                "covered": [a.render() for a in post.covered],
                "declined": [a.render() for a in post.declined],
                "appended": [a.render() for a in post.appended],
                "appended_levels": [a.level for a in post.appended],
                "doses": post.doses,
            })
    return rows


def summarize_e2e(rows: list[dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == cond]
        checks_raw = [c for r in rs for c in r["checks_raw"]]
        checks_final = [c for r in rs for c in r["checks_final"]]
        covered = sum(len(r["covered"]) + len(r["declined"]) for r in rs)
        appended = sum(len(r["appended"]) for r in rs)
        serious = sum(1 for r in rs for lvl in r["appended_levels"] if LEVEL_RANK[lvl] >= LEVEL_RANK["avoid"])
        out[cond] = {
            "answers": len(rs),
            "checks_passed_raw": f"{sum(c['passed'] for c in checks_raw)}/{len(checks_raw)}",
            "checks_passed_final": f"{sum(c['passed'] for c in checks_final)}/{len(checks_final)}",
            "warnings_covered_by_model": covered,
            "warnings_appended_by_code": appended,
            "model_coverage": round(covered / (covered + appended), 3) if covered + appended else 1.0,
            "answers_with_appended": sum(1 for r in rs if r["appended"]),
            "appended_avoid_or_worse": serious,
            "answers_with_doses": sum(1 for r in rs if r.get("doses")),
            "clarify_asked_by_model": f"{sum(len(r['clarify']['asked']) for r in rs if r.get('clarify'))}/"
                                      f"{sum(len(r['clarify']['asked']) + len(r['clarify']['appended']) for r in rs if r.get('clarify'))}",
            "answers_with_appended_questions": sum(1 for r in rs if r.get("clarify", {}).get("appended")),
        }
    return out


# =========================================================================== справочник ГРЛС: что изменилось


def normalization_diff(seed: Normalizer, full: Normalizer, golden: GoldenSet,
                       dev_dialogs: list[dict] | None = None, cases: list[dict] | None = None) -> dict[str, Any]:
    """Что меняет большой справочник по сравнению со стартовым.

    * names — названия препаратов и аллергенов из эталонной разметки: распознаны ли и в какой ключ;
    * mentions — упоминания препаратов в репликах покупателей, которые находит только большой справочник
      (сюда попадают и новые препараты, и ложные срабатывания на обычные слова — их стоит просмотреть).
    """
    from .sections import Section

    extractions, texts = [], []
    for _, _, s in golden.sessions():
        extractions.append(s["gold_extraction"])
        texts += [t["content"] for t in s["turns"] if t["role"] == "user"]
    for _, _, p in golden.probes():
        texts.append(p["text"])
    for d in dev_dialogs or []:
        extractions.append(d["gold_extraction"])
        texts += [t["content"] for t in d["turns"] if t["role"] == "user"]
    for c in cases or []:
        texts.append(c["question"])

    names: dict[tuple[str, str], dict] = {}
    for ex in extractions:
        for ev in ex.get("facts", []):
            if ev.get("section") not in ("medications", "allergies") or not ev.get("name"):
                continue
            sec = Section(ev["section"])
            a, b = seed.entity(sec, ev["name"]), full.entity(sec, ev["name"])
            names[(sec.value, ev["name"])] = {
                "section": sec.value, "name": ev["name"], "seed": a.normalized if a.matched else "—",
                "full": b.normalized if b.matched else "—", "changed": a.normalized != b.normalized,
            }
    extra: dict[str, dict] = {}
    for t in texts:
        known = {m.normalized for m in seed.drug_mentions(t)}
        for m in full.drug_mentions(t):
            if m.normalized not in known:
                row = extra.setdefault(m.normalized, {"key": m.normalized, "found_as": m.cleaned, "count": 0,
                                                      "example": t[:120]})
                row["count"] += 1
    rows = list(names.values())
    return {
        "names": len(rows),
        "matched_seed": sum(r["seed"] != "—" for r in rows),
        "matched_full": sum(r["full"] != "—" for r in rows),
        "changed": [r for r in rows if r["changed"]],
        "unmatched_full": sorted({r["name"] for r in rows if r["full"] == "—"}),
        "extra_mentions": sorted(extra.values(), key=lambda r: -r["count"]),
    }
