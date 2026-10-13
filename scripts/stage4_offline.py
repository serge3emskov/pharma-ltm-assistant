"""Этап 4 без GPU: проверочные случаи правил и повтор ответов этапа 3 через проверку после ответа.

Запуск: python scripts/stage4_offline.py [--grls path/to/catalog.json.gz]
Результат: reports/stage4_offline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharmaltm import DrugCatalog, Normalizer, SafetyChecker, __version__  # noqa: E402
from pharmaltm.evaluation import load_golden  # noqa: E402
from pharmaltm.safety_eval import load_cases, replay_answers, run_cases, summarize_cases  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grls", help="кэш справочника ГРЛС (DrugCatalog.save)")
    ap.add_argument("--stage3", default=str(ROOT / "reports" / "stage3_run2.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "stage4_offline.json"))
    args = ap.parse_args()

    catalog = DrugCatalog.load(args.grls) if args.grls else DrugCatalog()
    checker = SafetyChecker(catalog)
    normalizer = Normalizer(catalog=catalog)

    rows = run_cases(checker, load_cases(), normalizer)
    summary = summarize_cases(rows)
    print("Проверочные случаи:", json.dumps(summary, ensure_ascii=False))

    stage3 = json.loads(Path(args.stage3).read_text(encoding="utf-8"))
    replay = replay_answers(stage3, load_golden(), checker, normalizer)
    by_model: dict[str, dict] = {}
    for r in replay:
        m = by_model.setdefault(r["model"], {"answers": 0, "with_appended": 0, "appended": 0, "covered": 0,
                                             "declined": 0})
        m["answers"] += 1
        m["with_appended"] += bool(r["appended"])
        m["appended"] += len(r["appended"])
        m["covered"] += len(r["covered"])
        m["declined"] += len(r["declined"])
    print("Повтор ответов этапа 3:", json.dumps(by_model, ensure_ascii=False))
    for r in replay:
        if r["appended"]:
            print(f"\n[{r['model']}] {r['scenario']}: {r['question']}\n  ответ: {r['answer'][:160]}…")
            for a in r["appended"]:
                print("  + " + a)

    out = {
        "created": datetime.now().isoformat(timespec="seconds"), "version": __version__,
        "catalog": catalog.summary(), "rules": {"cases": summary, "rows": rows},
        "replay": {"source": Path(args.stage3).name, "by_model": by_model, "rows": replay},
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
