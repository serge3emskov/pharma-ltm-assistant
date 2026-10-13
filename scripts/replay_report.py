"""Проигрывает записанные ответы модели из отчёта этапа 3 через текущий код write-path.

Модель не нужна: берутся ответы extractor-а из упавших сценариев (`failures` в отчёте)
и заново проходят валидацию, проверки согласованности и state machine. Так видно,
сколько ошибок закрывают изменения в коде на тех же ответах модели. Отдельно
отмечаются сценарии, где текущая страховка маршрутизации открыла бы раздел,
для которого в отчёте нет ответа модели, — их проверит только новый прогон.

Запуск: python scripts/replay_report.py reports/stage3_run1.json
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharmaltm import MemoryStore, MemoryWriter, Normalizer, Section  # noqa: E402
from pharmaltm.evaluation import START, USER, _snapshot, load_golden  # noqa: E402
from pharmaltm.pipeline import WritePath  # noqa: E402


def replay(report: dict, golden_by_id: dict, normalizer: Normalizer) -> dict:
    probe = WritePath(None, MemoryWriter(MemoryStore(), normalizer))
    fixed, routing, remaining = [], [], []
    for failure in report["failures"]:
        scenario = golden_by_id[failure["scenario"]]
        steps = [s for s in scenario["steps"] if s["type"] == "session"]
        now = [START]
        store = MemoryStore()
        writer = MemoryWriter(store, normalizer, lambda: now[0])
        needs_routing = False
        for rec, step in zip(failure["sessions"], steps):
            now[0] += timedelta(days=step.get("after_days", 0))
            if step.get("setup"):
                writer.write(USER, step["setup"], "setup")
            writer.write(USER, rec["pred_extraction"], "replay",
                         dialog_text="\n".join(t["content"] for t in rec["turns_seen"]))
            routed = [Section(x) for x in set(rec["classifier_sections"]) | set(rec["safety_net"])]
            opened = {s.value for s in probe.safety_net(rec["turns_seen"], routed)}
            needs_routing |= bool(opened & set(step["gold_sections"]))
        facts, states = _snapshot(store)
        expect = steps[-1]["expect"]
        ok = facts == {tuple(x) for x in expect["facts"]} and states == {tuple(x) for x in expect["states"]}
        (fixed if ok else routing if needs_routing else remaining).append(failure["scenario"])
    return {"failed": len(report["failures"]), "fixed_by_code": fixed, "needs_new_routing": routing,
            "remaining": remaining}


def main() -> None:
    bundle = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    reports = bundle.get("reports") or {bundle["model"]: bundle}
    golden_by_id = {sc["id"]: sc for sc in load_golden().scenarios}
    normalizer = Normalizer()
    for model, report in reports.items():
        r = replay(report, golden_by_id, normalizer)
        print(f"{model}: упало {r['failed']} → исправлено кодом {len(r['fixed_by_code'])}, "
              f"ждут новой маршрутизации {len(r['needs_new_routing'])}, осталось {len(r['remaining'])}")
        for key in ("fixed_by_code", "needs_new_routing", "remaining"):
            print(f"  {key}: {', '.join(r[key]) or '—'}")


if __name__ == "__main__":
    main()
