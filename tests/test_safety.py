"""Этап 4: проверка противопоказаний и взаимодействий в коде."""
from __future__ import annotations

import json

import pytest

from pharmaltm import MemoryStore, MemoryWriter, Normalizer, QueryType
from pharmaltm import reference_data as ref
from pharmaltm.llm import ScriptedLLM
from pharmaltm.pipeline import PharmacyAssistant
from pharmaltm.retrieval import ContextBuilder
from pharmaltm.safety import SAFETY_HEADER, RuleSet, SafetyChecker, split_sentences
from pharmaltm.safety_eval import (
    load_cases,
    run_cases,
    run_safety_e2e,
    summarize_cases,
    summarize_e2e,
)

USER = "u"


@pytest.fixture(scope="module")
def checker() -> SafetyChecker:
    return SafetyChecker()


def profile_of(checker, facts, subject="self", statuses=None):
    store = MemoryStore()
    writer = MemoryWriter(store, Normalizer(catalog=checker.catalog))
    rep = writer.write(USER, {"facts": facts, "section_statuses": statuses or []})
    assert not rep.rejected, rep.rejected
    return checker.profile(store, USER, subject)


def f(section, name, **kw):
    return {"section": section, "name": name, "evidence": kw.pop("evidence", name), **kw}


DRIVER = {"section": "lifestyle", "attribute": "driving", "value": "да", "evidence": "весь день за рулём"}


# ---------------------------------------------------------------- данные правил


def test_rules_load_and_reference_known_names():
    rules = RuleSet.load()
    conditions = {e.canonical for e in ref.CONDITIONS}
    for r in rules.rules:
        assert set(r.conditions) <= conditions, (r.id, r.conditions)  # ключи памяти, а не свободный текст
    used = {c for r in rules.rules for sel in (r.drug, r.other, r.allergen) if sel for c in sel.classes}
    unused = {c.id for c in rules.classes} - used - {c.id for c in rules.classes if c.allergy_keys}
    assert not unused, f"классы без правил: {unused}"


def test_rule_validation_errors(tmp_path):
    data = json.loads((RuleSet.load.__globals__["DATA_DIR"] / "rules.json").read_text(encoding="utf-8"))
    classes = (RuleSet.load.__globals__["DATA_DIR"] / "classes.json").read_text(encoding="utf-8")
    data["rules"].append({"id": "bad", "kind": "condition", "drug": {"classes": ["nope"]}, "conditions": ["астма"],
                          "level": "caution", "message": "m", "reason": "x"})
    (tmp_path / "rules.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "classes.json").write_text(classes, encoding="utf-8")
    with pytest.raises(ValueError, match="неизвестные классы"):
        RuleSet.load(tmp_path)


# ---------------------------------------------------------------- классы препаратов


@pytest.mark.parametrize("name, has, has_not", [
    ("Нурофен", {"nsaid"}, {"antiplatelet"}),
    ("Вольтарен Эмульгель", set(), {"nsaid"}),                 # гель: местная форма
    ("Терафлю", {"antihistamine_sedating", "decongestant_systemic", "paracetamol"}, {"decongestant_nasal"}),
    ("Виброцил", {"decongestant_nasal"}, {"decongestant_systemic", "antihistamine_sedating"}),
    ("Кардиомагнил", {"nsaid", "antiplatelet"}, {"chelating"}),
    ("Смекта", {"sorbent", "chelating"}, set()),
    ("Аспирин Кардио", {"antiplatelet", "nsaid"}, set()),
    ("Панангин", {"potassium"}, set()),
    ("Корвалол", {"cns_depressant"}, set()),
])
def test_drug_classes(checker, name, has, has_not):
    d = checker.resolve(name)
    assert has <= d.classes and not (has_not & d.classes), d.classes


# ---------------------------------------------------------------- проверочные случаи


def test_all_safety_cases(checker):
    rows = run_cases(checker, load_cases())
    failed = [(r["id"], r["missing"], r["extra"]) for r in rows if not r["ok"]]
    assert not failed
    s = summarize_cases(rows)
    assert s["cases"] >= 40 and s["negative_controls"] >= 5 and s["precision"] == s["recall"] == 1.0


def test_cases_cover_every_kind_of_rule():
    cases = load_cases()
    rules = {r.id: r.kind for r in RuleSet.load().rules}
    kinds = {rules.get(rid.split(":", 1)[-1], rid) for c in cases for rid, _ in c["expect"]}
    for kind in ("drug_drug", "condition", "special_state", "age", "lifestyle", "allergy_cross",
                 "duplicate_ingredient", "allergy_class", "allergy_direct", "intolerance"):
        assert kind in kinds, kind
    assert sum(1 for c in cases if c["answer_checks"]) >= 20


# ---------------------------------------------------------------- профиль


def test_profile_takes_only_current_facts(checker):
    p = profile_of(checker, [
        f("medications", "Энап", status="stopped"), f("medications", "Лозап 50"),
        f("allergies", "пенициллин"), f("allergies", "ибупрофен", polarity="negative"),
        f("chronic_conditions", "гастрит", status="remission"), f("special_states", "беременность", polarity="negative"),
        {"section": "demographics", "attribute": "age", "value": 8, "evidence": "8 лет"}, DRIVER,
    ])
    assert [m.canonical for m in p.medications] == ["лозартан"]
    assert [a.key for a in p.allergies] == ["пенициллины"] and p.allergies[0].class_ids == {"penicillin"}
    assert p.conditions == {"гастрит": "remission"} and p.states == {}
    assert p.age == 8 and p.child and p.lifestyle == {"driving": "да"}


def test_condition_in_remission_is_named(checker):
    p = profile_of(checker, [f("chronic_conditions", "язва желудка", status="remission")])
    alerts = checker.check([checker.resolve("Нурофен")], p)
    assert alerts[0].trigger == "язвенная болезнь желудка (ремиссия)"


# ---------------------------------------------------------------- проверка до ответа


def test_report_restrictions_only_for_open_questions(checker):
    p = profile_of(checker, [DRIVER])
    rep = checker.report(p, "Можно Супрастин?")
    assert [a.rule_id for a in rep.alerts] == ["sedating+driving"] and rep.restrictions == []
    lines = rep.render()
    assert lines[1] == SAFETY_HEADER and "Препараты из вопроса:" in lines and "Ограничения по профилю (чего не предлагать):" not in lines
    open_q = checker.report(p, "Что-нибудь от аллергии?")
    assert open_q.alerts == [] and {r.rule_id for r in open_q.restrictions} >= {"sedating+driving", "cns+driving"}
    assert "Ограничения по профилю (чего не предлагать):" in open_q.render()

def test_rx_only_rules_are_not_profile_hints(checker):
    p = profile_of(checker, [f("special_states", "беременность")])
    targets = {r.target for r in checker.restrictions(p, limit=50)}
    assert "фторхинолоны, тетрациклины" not in targets and "статины" not in targets
    assert "НПВП" in targets
    # для препарата, названного покупателем, рецептурное правило работает
    assert checker.check([checker.resolve("Ципролет")], p)[0].rule_id == "fluoroquinolone_tetracycline+pregnancy"


def test_alternatives_are_not_a_combination(checker):
    assert not checker.is_combination("Что лучше от головы — Нурофен или Найз?")
    assert checker.is_combination("Можно ли пить Нурофен вместе с Найзом?")
    assert checker.is_combination("Можно Нурофен и Найз?")


def test_no_restrictions_for_non_recommending_queries(checker):
    store = MemoryStore()
    n = Normalizer(catalog=checker.catalog)
    MemoryWriter(store, n).write(USER, {"facts": [f("chronic_conditions", "язва желудка")]})
    b = ContextBuilder(store, n, safety=checker)
    assert b.build(USER, [QueryType.THERAPY_HISTORY]).safety.restrictions == []
    assert b.build(USER, [QueryType.SYMPTOM_RELIEF]).safety.restrictions
    general = b.build(USER, [QueryType.GENERAL], reference_text="Есть Нурофен в наличии?")
    assert "ПРОВЕРКА ПО СПРАВОЧНИКУ" in general.render() and "язвенная болезнь" in general.render()


def test_builder_without_safety_renders_as_before(checker):
    store = MemoryStore()
    n = Normalizer(catalog=checker.catalog)
    MemoryWriter(store, n).write(USER, {"facts": [f("chronic_conditions", "язва желудка")]})
    plain = ContextBuilder(store, n).build(USER, [QueryType.SYMPTOM_RELIEF], reference_text="Можно Нурофен?")
    assert plain.safety is None and "ПРОВЕРКА" not in plain.render()


# ---------------------------------------------------------------- проверка после ответа


@pytest.fixture(scope="module")
def ulcer_warfarin(checker):
    return profile_of(checker, [f("chronic_conditions", "язва желудка"), f("medications", "Варфарин")])


def test_post_check_appends_missing_warning(checker, ulcer_warfarin):
    post = checker.post_check("Возьмите Нурофен, он хорошо снимает боль.", ulcer_warfarin)
    assert {a.rule_id for a in post.appended} == {"nsaid+ulcer", "nsaid+anticoagulant"}
    assert post.text.startswith(post.raw) and "⚠ Проверка по справочнику:" in post.text
    assert "Нурофен (ибупрофен) + варфарин (уже принимается)" in post.text


def test_post_check_reason_must_be_next_to_drug(checker, ulcer_warfarin):
    answer = "Учитываю вашу язву. Можно принять Нурофен. Варфарин продолжайте как назначено."
    post = checker.post_check(answer, ulcer_warfarin)
    assert {a.rule_id for a in post.appended} == {"nsaid+ulcer", "nsaid+anticoagulant"}
    answer = "Нурофен при язве нельзя — риск кровотечения, тем более вместе с варфарином."
    post = checker.post_check(answer, ulcer_warfarin)
    assert not post.appended and len(post.covered) == 2 and post.text == answer


def test_post_check_declined_without_reason(checker, ulcer_warfarin):
    post = checker.post_check("Нурофен вам лучше не принимать. Возьмите Парацетамол.", ulcer_warfarin)
    assert not post.appended and {a.rule_id for a in post.declined} == {"nsaid+ulcer", "nsaid+anticoagulant"}


def test_post_check_question_drug_not_named_in_answer(checker, ulcer_warfarin):
    q = checker.mentions("Можно Найз?", "question")
    post = checker.post_check("При язве и приёме варфарина этот препарат не подходит из-за риска кровотечения. "
                              "Возьмите Парацетамол.", ulcer_warfarin, q)
    assert not post.appended and {a.drug.canonical for a in post.covered} == {"нимесулид"}
    post = checker.post_check("Возьмите Парацетамол.", ulcer_warfarin, q)
    assert {a.drug.canonical for a in post.appended} == {"нимесулид"}


def test_post_check_ignores_alternatives_in_answer(checker):
    p = profile_of(checker, [])
    post = checker.post_check("Подойдёт Нурофен или Найз — выберите одно.", p)
    assert post.alerts == [] and post.text == post.raw


def test_post_check_answer_drug_vs_drug_from_question(checker):
    p = profile_of(checker, [])
    q = checker.mentions("Я уже выпил Терафлю, что ещё от головы?", "question")
    post = checker.post_check("Можно Панадол.", p, q)
    assert [a.rule_id for a in post.appended] == ["duplicate_ingredient"]


def test_post_check_names_subject_and_skips_info(checker):
    p = profile_of(checker, [{"section": "demographics", "subject": "мама", "attribute": "age", "value": 80,
                              "evidence": "маме 80"}], subject="мама")
    post = checker.post_check("Маме подойдёт Донормил.", p)
    assert "⚠ Проверка по справочнику (для: мама):" in post.text
    assert checker.post_check("Маме подойдёт Донормил.", p, min_level="contraindicated").appended == []


def test_split_sentences():
    assert split_sentences("Раз. Два! Три?\nЧетыре") == ["Раз.", "Два!", "Три?", "Четыре"]


# ---------------------------------------------------------------- ассистент


def _bot(answer: str, safety=True):
    def respond(m, meta):
        task = meta["task"]
        if task == "classify_query":
            return json.dumps({"query_types": ["symptom_relief"], "subject": "self"})
        if task == "classify_sections":
            return json.dumps({"sections": []})
        return answer
    llm = ScriptedLLM(respond)
    bot = PharmacyAssistant(llm, safety=safety)
    bot.writer.write(USER, {"facts": [DRIVER]})
    return bot, llm


def test_assistant_appends_warning_and_keeps_raw():
    bot, llm = _bot("Возьмите Супрастин.")
    reply = bot.reply(USER, "Аллергия на цветение, что взять? Может, Супрастин?")
    system = [c for c in llm.calls if c["meta"]["task"] == "answer"][-1]["messages"][0]["content"]
    assert "ПРОВЕРКА ПО СПРАВОЧНИКУ" in system and "вождение автомобиля" in system
    assert reply.raw_text == "Возьмите Супрастин." and reply.text != reply.raw_text
    assert reply.safety.appended[0].rule_id == "sedating+driving"
    assert reply.as_dict()["safety"]["appended"][0]["rule"] == "sedating+driving"


def test_assistant_without_safety_is_stage3():
    bot, llm = _bot("Возьмите Супрастин.", safety=False)
    reply = bot.reply(USER, "Аллергия на цветение, что взять? Может, Супрастин?")
    system = [c for c in llm.calls if c["meta"]["task"] == "answer"][-1]["messages"][0]["content"]
    assert "ПРОВЕРКА" not in system and reply.text == reply.raw_text == "Возьмите Супрастин." and reply.safety is None


def test_session_history_keeps_final_text():
    bot, _ = _bot("Возьмите Супрастин.")
    s = bot.session(USER)
    text = s.say("Можно Супрастин?")
    assert s.turns[-1]["content"] == text and "⚠" in text


# ---------------------------------------------------------------- E2E-обвязка


def test_run_safety_e2e_with_scripted_model(checker):
    cases = [c for c in load_cases() if c["id"] in ("s01", "s05")]
    prompts = []

    def respond(m, meta):
        prompts.append(m[0]["content"])
        return "Супрастин вызывает сонливость, за руль лучше не садиться." if "Супрастин" in m[-1]["content"] \
            else "Можно Нурофен."

    rows = run_safety_e2e(ScriptedLLM(respond), checker, cases)
    assert [(r["id"], r["condition"]) for r in rows] == [("s01", "baseline"), ("s05", "baseline"),
                                                         ("s01", "safety"), ("s05", "safety")]
    assert "ПРОВЕРКА" not in prompts[0] and "ПРОВЕРКА" in prompts[2]
    s = summarize_e2e(rows)
    assert s["safety"]["warnings_covered_by_model"] == 1 and s["safety"]["warnings_appended_by_code"] == 1
    assert s["safety"]["checks_passed_raw"] == "1/2" and s["safety"]["checks_passed_final"] == "2/2"


# ---------------------------------------------------------------- golden set с проверкой


from pharmaltm.catalog import DrugCatalog, Product  # noqa: E402
from pharmaltm.devcheck import GoldLLM  # noqa: E402
from pharmaltm.evaluation import gold_backend, load_golden, run_golden, summarize, to_report  # noqa: E402
from pharmaltm.safety_eval import replay_answers  # noqa: E402


@pytest.fixture(scope="module")
def golden():
    return load_golden()


def test_golden_with_safety_and_post_check(checker, golden):
    llm = gold_backend(golden)
    llm.answer_text = "Возьмите Супрастин."
    run = run_golden(llm, golden, safety=checker)
    s = summarize(run)["e2e_answers"]
    assert s["safety_checked"] == s["probes"] > 0 and s["answers_with_appended_warnings"] >= 1
    rep = to_report(run, golden)
    driver = next(p for p in rep["probes"] if p["scenario"] == "g-e2e-09")
    assert driver["answer_raw"] == "Возьмите Супрастин." and "вождение автомобиля" in driver["answer"]
    assert driver["post_check"]["appended"][0]["rule"] == "sedating+driving"


def test_golden_without_safety_keeps_stage3_prompts(golden):
    run = run_golden(gold_backend(golden), golden)
    probes = [p for r in run.scenarios for p in r.probes if p.answer_prompt]
    assert probes and all("ПРОВЕРКА" not in p.answer_prompt[0]["content"] for p in probes)
    assert all(p.answer == p.answer_raw and p.post_check is None for p in probes)


def test_replay_stage3_answers(checker, golden):
    llm = gold_backend(golden)
    llm.answer_text = "Подойдёт Супрастин или Цетрин."
    report = to_report(run_golden(llm, golden), golden)
    rows = replay_answers(report, golden, checker)
    driver = next(r for r in rows if r["scenario"] == "g-e2e-09")
    assert {a.split(" — ")[0] for a in driver["appended"]} == {"Супрастин (хлоропирамин)", "Цетрин (цетиризин)"}


def test_golden_memory_unchanged_with_grls_catalog(golden):
    """Наименования из ГРЛС, совпадающие со стартовым набором, не меняют ключи памяти golden set."""
    cat = DrugCatalog()
    cat.add_products([
        Product("Энап", ("эналаприл",), ("C09AA02",)), Product("Найз", ("нимесулид",), ("M01AX17",)),
        Product("ТераФлю", ("парацетамол", "фенирамин", "фенилэфрин"), ("N02BE51",)),
        Product("Супрастин", ("хлоропирамин",), ("R06AC03",)), Product("Лозап", ("лозартан",), ("C09CA01",)),
        Product("Ингавирин", ("имидазолилэтанамид пентандиовой кислоты",), ("J05AX",)),
    ])
    s = summarize(run_golden(gold_backend(golden), golden, normalizer=Normalizer(catalog=cat), run_answers=False))
    assert s["memory"]["facts"]["f1"] == 1.0 and s["memory"]["section_states"]["f1"] == 1.0


@pytest.mark.parametrize("answer, doses", [
    ("ибупрофен (в небольших дозах, не более 200 мг за раз, не чаще 2 раз в сутки)", ["200 мг", "2 раз в сутки"]),
    ("По 1 таблетке 3 раза в день.", ["1 таблетке", "3 раза в день"]),
    ("Температура 38,5 — сверьтесь с инструкцией.", []),
])
def test_post_check_counts_doses(checker, answer, doses):
    assert checker.post_check(answer, profile_of(checker, [])).doses == doses


def test_restrictions_do_not_repeat_pair_alert(checker):
    p = profile_of(checker, [f("medications", "Варфарин")])
    alerts = checker.check(checker.mentions("Можно Найз?", "question"), p)
    said = {(a.rule_id, f"принимает {a.other.label}") for a in alerts if a.other}
    assert ("nsaid+anticoagulant", "принимает варфарин") in said
    rest = checker.report(p, "Что-нибудь от боли в спине?").restrictions
    assert {r.rule_id for r in rest} >= {"nsaid+anticoagulant", "azole+warfarin"}


# ---------------------------------------------------------------- справочник в формате выгрузки ГРЛС


@pytest.fixture(scope="module")
def real_checker(tmp_path_factory):
    from test_catalog import REAL_ROWS, _real_xlsx

    d = tmp_path_factory.mktemp("grls_safety")
    (d / "grls-Действующий.xlsx").write_bytes(_real_xlsx("Действующий", REAL_ROWS))
    return SafetyChecker(DrugCatalog.from_grls(d)[0])


@pytest.mark.parametrize("name, has, has_not", [
    ("Фастум гель", set(), {"nsaid"}),                          # без АТХ: местную форму видно по «Формам выпуска»
    ("Нурофен", {"nsaid"}, set()),                              # таблетки и гель под одним наименованием
    ("Валсакор", {"arb", "antihypertensive"}, set()),            # класс по МНН, когда АТХ нет
    ("Метипред", {"corticosteroid_systemic"}, set()),
    ("Тридерм", set(), {"corticosteroid_systemic"}),
    ("Визин", set(), {"decongestant_nasal"}),                    # глазные капли
    ("Лиотон 1000", set(), {"anticoagulant"}),
])
def test_classes_without_atc(real_checker, name, has, has_not):
    d = real_checker.resolve(name)
    assert d.info.matched and has <= d.classes and not (has_not & d.classes), (d.info, d.classes)


def test_topical_nsaid_with_ulcer_has_no_alert(real_checker):
    p = profile_of(real_checker, [f("chronic_conditions", "язва желудка")])
    assert real_checker.report(p, "Болит колено, Фастум гель можно?").alerts == []
    assert [a.rule_id for a in real_checker.report(p, "А Нурофен?").alerts] == ["nsaid+ulcer"]


def test_rx_only_drug_in_answer(real_checker):
    p = profile_of(real_checker, [f("medications", "Метипред")])
    post = real_checker.post_check("Можно взять Цефтриаксон или Омез.", p)
    assert [(a.rule_id, a.drug.canonical) for a in post.appended] == [("rx_only", "цефтриаксон")]
    assert "Цефтриаксон — по ГРЛС отпускается только по рецепту" in post.text
    covered = real_checker.post_check("Цефтриаксон отпускается по рецепту, нужен врач.", p)
    assert not covered.appended and covered.covered[0].rule_id == "rx_only"
    # назначенный препарат из памяти — не повод напоминать о рецепте
    assert real_checker.report(p, "Метипред можно пить утром?").alerts == []


# ---------------------------------------------------------------- находки прогона на реальной выгрузке ГРЛС


def test_composition_mentions_are_not_new_drugs(checker):
    p = profile_of(checker, [f("special_states", "беременность")])
    q = checker.mentions("Простыла, можно выпить Терафлю?", "question")
    post = checker.post_check("Препарат содержит парацетамол, фенилэфрин и фенирамин. Терафлю при беременности "
                              "противопоказан.", p, q)
    assert {a.drug.canonical for a in post.alerts} == {"парацетамол + фенилэфрин + фенирамин"}
    assert [d.canonical for d in post.answer_drugs] == ["парацетамол + фенилэфрин + фенирамин"]


def test_inn_of_taken_drug_is_a_reference(checker):
    p = profile_of(checker, [f("medications", "Эутирокс"), f("medications", "Кардиомагнил")])
    q = checker.mentions("Изжога, можно Алмагель?", "question")
    post = checker.post_check("Вы принимаете левотироксин, поэтому Алмагель — с интервалом 4 часа. "
                              "Кардиомагнил (ацетилсалициловая кислота) продолжайте.", p, q)
    assert [a.rule_id for a in post.alerts] == ["levothyroxine+chelating"] and not post.appended


def test_trade_name_duplicate_still_caught(checker):
    p = profile_of(checker, [f("medications", "МИГ")])
    post = checker.post_check("Можно принять Нурофен.", p)
    assert [a.rule_id for a in post.appended] == ["duplicate_ingredient"]


def test_coverage_by_name_stem(checker):
    p = profile_of(checker, [{"section": "lifestyle", "attribute": "alcohol", "value": "да", "evidence": "выпиваю"}])
    q = checker.mentions("Голова болит, Панадол можно?", "question")
    post = checker.post_check("Можно. Парацетамоловые средства при алкоголе вредят печени.", p, q)
    assert not post.appended and post.covered[0].rule_id == "paracetamol+alcohol"


def test_rx_only_skips_prescribed_and_known(real_checker):
    p = profile_of(real_checker, [f("medications", "Метипред", status="stopped")])
    assert real_checker.report(p, "Врач выписал Цефтриаксон, где купить?").alerts == []
    assert real_checker.report(p, "Чем заменить Метипред?").alerts == []   # принимал раньше — назначал врач
    assert [a.rule_id for a in real_checker.report(p, "Можно Цефтриаксон?").alerts] == ["rx_only"]


def test_genitive_trade_name_without_inn_is_merged():
    cat = DrugCatalog()
    cat.add_products([Product("Парацетамола таблетки", (), (), routes=("systemic",), otc=True)])
    assert cat.resolve("Парацетамола").canonical == "парацетамол"
    assert [m.canonical for m in cat.mentions("при употреблении парацетамола")] == ["парацетамол"]
    assert cat.summary()["merged_inn_variants"] == 1 and "парацетамола" not in cat


def test_reason_in_next_sentence_about_same_drug(checker):
    p = profile_of(checker, [DRIVER])
    q = checker.mentions("Аллергия на цветение — Супрастин подойдёт?", "question")
    ok = checker.post_check("Супрастин помогает при аллергии. Учитывая, что вы за рулём, он вызывает сонливость.", p, q)
    assert not ok.appended
    other = checker.post_check("Супрастин помогает при аллергии. Цетрин вызывает сонливость реже.", p, q)
    assert [a.rule_id for a in other.appended] == ["sedating+driving"]  # про Цетрин причина названа


def test_contraindication_must_be_a_prohibition(checker):
    p = profile_of(checker, [f("chronic_conditions", "язва желудка")])
    q = checker.mentions("Болит голова, Нурофен подойдёт?", "question")
    soft = checker.post_check("Нурофен может помочь. Учитывайте язву — НПВП могут её усугубить.", p, q)
    assert [a.rule_id for a in soft.appended] == ["nsaid+ulcer"]
    hard = checker.post_check("Нурофен при язве противопоказан — возьмите Парацетамол.", p, q)
    assert not hard.appended


# ---------------------------------------------------------------- прогон 2 на GPU: что исправлено


BPH = f("chronic_conditions", "аденома простаты")


def test_not_contraindicated_is_not_a_warning(checker):
    p = profile_of(checker, [BPH])
    q = checker.mentions("Аллергия, можно Супрастин?", "question")
    post = checker.post_check("При аденоме простаты Супрастин не противопоказан, но может вызывать сонливость.", p, q)
    assert [a.rule_id for a in post.appended] == ["sedating+bph"] and not post.declined
    ok = checker.post_check("При аденоме простаты Супрастин не рекомендуется без врача. "
                            "Препарат может вызвать задержку мочи.", p, q)
    assert not ok.appended and [a.rule_id for a in ok.covered] == ["sedating+bph"]


def test_decline_must_be_about_the_new_drug(checker):
    p = profile_of(checker, [f("medications", "Кардиомагнил")])
    q = checker.mentions("Болит голова, можно Нурофен?", "question")
    answer = ("Да, можно использовать Нурофен. Вы принимаете Кардиомагнил, поэтому избегайте других препаратов "
              "с ацетилсалициловой кислотой.")
    post = checker.post_check(answer, p, q)
    assert [a.rule_id for a in post.appended] == ["nsaid+antiplatelet"] and not post.declined


def test_restrictions_grouped_by_trigger(checker):
    p = profile_of(checker, [{"section": "demographics", "attribute": "age", "value": 6, "evidence": "6 лет"},
                             f("allergies", "Амоксиклав")])
    lines = checker.report(p, "Что дать от температуры?").render()
    age = [x for x in lines if x.startswith("- Возраст 6 — противопоказано:")]
    assert len(age) == 1 and "нимесулид — детям до 12 лет" in age[0] and "кеторолак" in age[0]
    allergy = [x for x in lines if x.startswith("- Аллергия: амоксициллин")]
    assert len(allergy) == 1 and "другие пенициллины — возможна перекрёстная реакция" in allergy[0]


def test_clarify_block_reminds_that_check_sees_only_memory(checker):
    store = MemoryStore()
    n = Normalizer(catalog=checker.catalog)
    MemoryWriter(store, n).write(USER, {"facts": [f("chronic_conditions", "язва желудка")]})
    with_safety = ContextBuilder(store, n, safety=checker).build(USER, [QueryType.SYMPTOM_RELIEF]).render()
    plain = ContextBuilder(store, n).build(USER, [QueryType.SYMPTOM_RELIEF]).render()
    assert "НУЖНО УТОЧНИТЬ" in with_safety and "эти вопросы не заменяет" in with_safety
    assert "НУЖНО УТОЧНИТЬ" in plain and "эти вопросы не заменяет" not in plain


def test_analog_question_gets_profile_restrictions(checker):
    p = profile_of(checker, [f("allergies", "Флемоксин")])
    rep = checker.report(p, "Чем заменить Флемоксин?")
    assert [a.rule_id for a in rep.alerts] == ["allergy_direct"]
    # модель выберет замену сама: ей нужно знать, что и цефалоспорины без врача не подходят
    assert [r.rule_id for r in rep.restrictions] == ["allergy_cross", "penicillin_allergy+cephalosporin"]
    assert checker.report(p, "Можно Флемоксин?").restrictions == []
    assert checker.report(p, "Флемоксин дорогой", choosing=True).restrictions  # классификатор: ANALOG


# ---------------------------------------------------------------- прогон 3: уточняющие вопросы


from pharmaltm.followup import clarify_check  # noqa: E402
from pharmaltm.retrieval import ClarifyItem  # noqa: E402
from pharmaltm.sections import Section  # noqa: E402

MEDS_CHRONIC = [ClarifyItem(Section.MEDICATIONS, "unknown"), ClarifyItem(Section.CHRONIC_CONDITIONS, "unknown")]


def test_clarify_questions_appended_when_model_recommends_without_asking():
    answer = ("Предлагаю ибупрофен или парацетамол. Учтите, что у сына нет данных о других лекарствах "
              "или хронических заболеваниях.")
    c = clarify_check(answer, MEDS_CHRONIC, "сын")
    assert c.asked == [] and c.questions == ["Принимает ли сын сейчас другие лекарства?",
                                             "Есть ли хронические заболевания?"]
    assert c.text.startswith(answer) and "уточните (для: сын)" in c.text


def test_clarify_question_already_asked():
    c = clarify_check("Подойдёт парацетамол. Уточните, есть ли хронические заболевания? А лекарства принимаете?",
                      MEDS_CHRONIC)
    assert len(c.asked) == 2 and c.appended == [] and c.text.endswith("принимаете?")


def test_no_clarify_without_recommendation():
    c = clarify_check("Спасибо, запомнил. Чем помочь?", MEDS_CHRONIC, recommended=False)
    assert c.appended == [] and c.text == "Спасибо, запомнил. Чем помочь?"


def test_assistant_appends_clarify_questions():
    bot, _ = _bot("Возьмите Парацетамол.")
    reply = bot.reply(USER, "Что-нибудь от головной боли?")
    assert reply.raw_text == "Возьмите Парацетамол." and reply.clarify is not None
    assert reply.clarify.appended and "❓" in reply.text
