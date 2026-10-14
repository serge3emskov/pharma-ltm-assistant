"""Пересчёт E2E этапа 4 по сохранённым ответам: текущая проверка после ответа, без новой генерации.

Запуск: python scripts/recheck_stage4.py reports/stage4_run2.json
Справочник — стартовый набор, поэтому проверка «только по рецепту» не участвует (она требует ГРЛС).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharmaltm import Normalizer, SafetyChecker  # noqa: E402
from pharmaltm.evaluation import check_answer  # noqa: E402
from pharmaltm.safety_eval import NOW, USER, case_store, load_cases  # noqa: E402


def main(path: str) -> None:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    checker = SafetyChecker()
    normalizer = Normalizer(catalog=checker.catalog)
    cases = {c["id"]: c for c in load_cases()}
    totals: dict[str, dict[str, int]] = {}
    for row in report["e2e"]["rows"]:
        case = cases[row["id"]]
        store = case_store(case, normalizer)
        profile = checker.profile(store, USER, normalizer.subject(case["subject"]), NOW)
        pre = checker.report(profile, case["question"])
        post = checker.post_check(row["raw"], profile, pre.question_drugs)
        t = totals.setdefault(row["condition"], {"answers": 0, "checks_raw": 0, "checks_final": 0, "handled": 0,
                                                 "appended": 0, "answers_with_appended": 0})
        t["answers"] += 1
        t["checks_raw"] += all(c["passed"] for c in check_answer(row["raw"], case["answer_checks"]))
        t["checks_final"] += all(c["passed"] for c in check_answer(post.text, case["answer_checks"]))
        t["handled"] += len(post.covered) + len(post.declined)
        t["appended"] += len(post.appended)
        t["answers_with_appended"] += bool(post.appended)
        for a in post.appended:
            print(f"{row['id']:>4} {row['condition']:<8} + {a.render()}")
    print()
    for cond, t in totals.items():
        share = t["handled"] / (t["handled"] + t["appended"]) if t["handled"] + t["appended"] else 1.0
        print(f"{cond}: эвристики {t['checks_raw']}/{t['answers']} → {t['checks_final']}/{t['answers']}; "
              f"учтено моделью {t['handled']} из {t['handled'] + t['appended']} ({share:.0%}); "
              f"ответов с дописанным предупреждением {t['answers_with_appended']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "reports" / "stage4_run2.json"))
