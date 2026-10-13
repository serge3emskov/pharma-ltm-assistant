"""Собирает golden set этапа 3: data/golden/scenarios.jsonl и data/golden/queries.jsonl.

Golden set — отложенная выборка: по ней промпты НЕ настраиваются (для этого есть
dev-набор). Каждый сценарий — последовательность шагов на одной временной оси:

* session — фрагмент диалога. Эталон записан дважды: ответ extractor-а
  (gold_extraction) и ожидаемый снимок памяти после сессии (expect). Тест
  прогоняет эталонные ответы через весь пайплайн и требует совпадения снимка:
  так ловятся ошибки и в разметке, и в коде.
* probe — сообщение покупателя после сессий: эталонный тип запроса и что
  должно и не должно попасть в контекст, что ассистент должен уточнить,
  плюс эвристические проверки ответа.

Запуск: python scripts/build_golden.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "golden"

# --------------------------------------------------------------------------- реплики


def U(text: str) -> dict:
    return {"role": "user", "content": text}


def A(text: str) -> dict:
    return {"role": "assistant", "content": text}


# --------------------------------------------------------------------------- события extractor-а


def _entity(section: str, name: str, ev: str, subj: str, pol: str, **kw) -> dict:
    return {"section": section, "subject": subj, "name": name, "polarity": pol, **kw, "evidence": ev}


def med(name, ev, subj="self", pol="positive", **kw):
    return _entity("medications", name, ev, subj, pol, **kw)


def alg(name, ev, subj="self", pol="positive", **kw):
    return _entity("allergies", name, ev, subj, pol, **kw)


def chron(name, ev, subj="self", pol="positive", **kw):
    return _entity("chronic_conditions", name, ev, subj, pol, **kw)


def spec(name, ev, subj="self", pol="positive", **kw):
    return _entity("special_states", name, ev, subj, pol, **kw)


def sym(name, ev, subj="self", pol="positive", **kw):
    return _entity("symptoms", name, ev, subj, pol, **kw)


def diag(name, ev, subj="self", pol="positive", **kw):
    return _entity("reported_diagnoses", name, ev, subj, pol, **kw)


def dem(attr, value, ev, subj="self"):
    return {"section": "demographics", "subject": subj, "attribute": attr, "value": value, "evidence": ev}


def life(attr, value, ev, subj="self", **kw):
    return {"section": "lifestyle", "subject": subj, "attribute": attr, "value": value, **kw, "evidence": ev}


def st(section, status, ev, subj="self"):
    return {"section": section, "subject": subj, "status": status, "evidence": ev}


# --------------------------------------------------------------------------- ожидания


def F(section, subj, key, status="active"):
    return [section, subj, key, status]


def S(section, subj, state):
    return [section, subj, state]


# --------------------------------------------------------------------------- шаги


def session(turns, facts=(), statuses=(), expect=None, states=(), data=(), setup=None, after_days=0):
    events = list(facts) + list(statuses)
    return {
        "type": "session",
        "after_days": after_days,
        "setup": setup,
        "turns": list(turns),
        "gold_sections": sorted({e["section"] for e in events}),
        "gold_extraction": {"facts": list(facts), "section_statuses": list(statuses)},
        "expect": None if expect is None else {"facts": list(expect), "states": list(states), "data": list(data)},
    }


def probe(text, types, subject="self", include=(), exclude=(), clarify=(), no_clarify=(), checks=(), after_days=0):
    return {
        "type": "probe",
        "after_days": after_days,
        "text": text,
        "query_types": list(types),
        "subject": subject,
        "context_include": [list(x) for x in include],
        "context_exclude": [list(x) for x in exclude],
        "clarify_include": [list(x) + [None] * (3 - len(x)) for x in clarify],
        "clarify_exclude": list(no_clarify),
        "answer_checks": [{"kind": k, "pattern": p, "why": w} for k, p, w in checks],
    }


def mentions(pattern, why):
    return ("mentions", pattern, why)


def not_mentions(pattern, why):
    return ("not_mentions", pattern, why)


SCENARIOS: list[dict] = []


def scenario(sid, category, tags, steps, note=""):
    SCENARIOS.append({"id": sid, "category": category, "tags": list(tags), "note": note, "steps": list(steps)})


def single(sid, category, tags, turns, facts=(), statuses=(), expect=(), states=(), data=(), setup=None, note=""):
    scenario(sid, category, tags,
             [session(turns, facts, statuses, expect=expect, states=states, data=data, setup=setup)], note)


M, AL, CH, SP, DE, LI, SY, DI = ("medications", "allergies", "chronic_conditions", "special_states",
                                 "demographics", "lifestyle", "symptoms", "reported_diagnoses")

# =========================================================================== лекарства

single("g-med-01", "medications", ["multi_drug", "dosage"],
       [U("Принимаю Конкор 2,5 мг утром и Аторис 20 вечером.")],
       [med("Конкор 2,5 мг", "Принимаю Конкор 2,5 мг утром", status="active", frequency="утром"),
        med("Аторис 20", "Аторис 20 вечером", status="active", frequency="вечером")],
       expect=[F(M, "self", "бисопролол"), F(M, "self", "аторвастатин")])

single("g-med-02", "medications", ["typo", "trap_conditional"],
       [U("Нурафен пью иногда, когда голова болит.")],
       [med("Нурафен", "Нурафен пью иногда", status="active", frequency="иногда", indication="головная боль")],
       expect=[F(M, "self", "ибупрофен")],
       note="«когда голова болит» — условие приёма, а не текущая жалоба")

single("g-med-03", "medications", ["prescribed"],
       [U("Мне выписали Сумамед, но я его ещё не начинала пить.")],
       [med("Сумамед", "Мне выписали Сумамед, но я его ещё не начинала пить", status="prescribed")],
       expect=[F(M, "self", "азитромицин", "prescribed")])

single("g-med-04", "medications", ["frequency"],
       [U("Глюкофаж пью уже лет пять, утром и вечером по таблетке.")],
       [med("Глюкофаж", "Глюкофаж пью уже лет пять, утром и вечером по таблетке", status="active",
            frequency="утром и вечером по таблетке")],
       expect=[F(M, "self", "метформин")])

single("g-med-05", "medications", ["paused"],
       [U("Эутирокс сейчас не пью — врач сказал сделать перерыв на месяц.")],
       [med("Эутирокс", "Эутирокс сейчас не пью — врач сказал сделать перерыв на месяц", status="paused")],
       expect=[F(M, "self", "левотироксин натрия", "paused")])

single("g-med-06", "medications", ["stopped"],
       [U("Лозап бросил месяц назад, сам.")],
       [med("Лозап", "Лозап бросил месяц назад", status="stopped")],
       expect=[F(M, "self", "лозартан", "stopped")])

single("g-med-07", "medications", ["multi_drug", "stopped"],
       [U("Пью Варфарин 2,5 мг, а Кардиомагнил отменили.")],
       [med("Варфарин 2,5 мг", "Пью Варфарин 2,5 мг", status="active"),
        med("Кардиомагнил", "Кардиомагнил отменили", status="stopped")],
       expect=[F(M, "self", "варфарин"),
               F(M, "self", "ацетилсалициловая кислота + магния гидроксид", "stopped")])

single("g-med-08", "medications", ["dosage", "frequency"],
       [U("Ксарелто 20 мг раз в день, вечером.")],
       [med("Ксарелто 20 мг", "Ксарелто 20 мг раз в день, вечером", status="active",
            frequency="раз в день, вечером")],
       expect=[F(M, "self", "ривароксабан")])

single("g-med-09", "medications", ["out_of_lexicon"],
       [U("Пью Детралекс по две таблетки в день.")],
       [med("Детралекс", "Пью Детралекс по две таблетки в день", status="active", frequency="по две таблетки в день")],
       expect=[F(M, "self", "детралекс")])

single("g-med-10", "medications", ["case_form", "stopped"],
       [U("Уже месяц лечусь Нольпазой, до этого был Омез.")],
       [med("Нольпаза", "Уже месяц лечусь Нольпазой", status="active"),
        med("Омез", "до этого был Омез", status="stopped")],
       expect=[F(M, "self", "пантопразол"), F(M, "self", "омепразол", "stopped")])

single("g-med-11", "medications", ["negation"],
       [U("Не принимаю ни Аспирин, ни Кардиомагнил.")],
       [med("Аспирин", "Не принимаю ни Аспирин", pol="negative"),
        med("Кардиомагнил", "ни Кардиомагнил", pol="negative")],
       expect=[F(M, "self", "ацетилсалициловая кислота", "denied"),
               F(M, "self", "ацетилсалициловая кислота + магния гидроксид", "denied")])

single("g-med-12", "medications", ["multi_drug", "dosage"],
       [U("По утрам Эналаприл 10 мг, вечером Амлодипин 5 мг.")],
       [med("Эналаприл 10 мг", "По утрам Эналаприл 10 мг", status="active", frequency="по утрам"),
        med("Амлодипин 5 мг", "вечером Амлодипин 5 мг", status="active", frequency="вечером")],
       expect=[F(M, "self", "эналаприл"), F(M, "self", "амлодипин")])

# =========================================================================== аллергии

single("g-alg-01", "allergies", ["drug_allergy"],
       [U("У меня аллергия на Амоксиклав — отекло лицо.")],
       [alg("Амоксиклав", "У меня аллергия на Амоксиклав — отекло лицо", reaction_type="allergy", reaction="отёк лица")],
       expect=[F(AL, "self", "амоксициллин + клавулановая кислота")])

single("g-alg-02", "allergies", ["intolerance"],
       [U("Ибупрофен плохо переношу — болит желудок.")],
       [alg("Ибупрофен", "Ибупрофен плохо переношу — болит желудок", reaction_type="intolerance",
            reaction="боль в желудке")],
       expect=[F(AL, "self", "ибупрофен")])

single("g-alg-03", "allergies", ["non_drug", "case_form"],
       [U("Аллергия на кошек и на пыль.")],
       [alg("кошки", "Аллергия на кошек", reaction_type="allergy"),
        alg("пыль", "на пыль", reaction_type="allergy")],
       expect=[F(AL, "self", "шерсть животных"), F(AL, "self", "домашняя пыль")])

single("g-alg-04", "allergies", ["negation"],
       [U("На йод у меня аллергии нет, а вот на латекс есть.")],
       [alg("йод", "На йод у меня аллергии нет", pol="negative"),
        alg("латекс", "а вот на латекс есть", reaction_type="allergy")],
       expect=[F(AL, "self", "йод", "denied"), F(AL, "self", "латекс")])

single("g-alg-05", "allergies", ["confirmed_absent"],
       [U("Аллергии ни на что нет.")],
       statuses=[st(AL, "confirmed_absent", "Аллергии ни на что нет")],
       expect=[], states=[S(AL, "self", "confirmed_absent")])

single("g-alg-06", "allergies", ["dont_know"],
       [U("Про аллергию на лекарства не знаю, никогда не обращала внимания.")],
       statuses=[st(AL, "dont_know", "Про аллергию на лекарства не знаю")],
       expect=[], states=[S(AL, "self", "dont_know")])

single("g-alg-07", "allergies", ["family"],
       [U("У сына аллергия на цитрусовые, высыпает.")],
       [alg("цитрусовые", "У сына аллергия на цитрусовые, высыпает", subj="сын", reaction_type="allergy",
            reaction="сыпь")],
       expect=[F(AL, "сын", "цитрусовые")])

single("g-alg-08", "allergies", ["drug_allergy"],
       [U("На сульфаниламиды была сильная реакция — крапивница.")],
       [alg("сульфаниламиды", "На сульфаниламиды была сильная реакция — крапивница", reaction_type="allergy",
            reaction="крапивница")],
       expect=[F(AL, "self", "сульфаниламиды")])

single("g-alg-09", "allergies", ["negation", "drug_allergy"],
       [U("На цефалоспорины аллергия, а пенициллины нормально переношу.")],
       [alg("цефалоспорины", "На цефалоспорины аллергия", reaction_type="allergy"),
        alg("пенициллины", "а пенициллины нормально переношу", pol="negative")],
       expect=[F(AL, "self", "цефалоспорины"), F(AL, "self", "пенициллины", "denied")])

# =========================================================================== хронические заболевания

single("g-chr-01", "chronic", ["implicit_drug"],
       [U("Астма с детства, пользуюсь ингалятором.")],
       [chron("астма", "Астма с детства", status="active")],
       expect=[F(CH, "self", "бронхиальная астма")],
       note="ингалятор без названия — не факт раздела medications")

single("g-chr-02", "chronic", ["multi_fact"],
       [U("У меня мерцательная аритмия и гипотиреоз.")],
       [chron("мерцательная аритмия", "У меня мерцательная аритмия", status="active"),
        chron("гипотиреоз", "и гипотиреоз", status="active")],
       expect=[F(CH, "self", "фибрилляция предсердий"), F(CH, "self", "гипотиреоз")])

single("g-chr-03", "chronic", ["remission"],
       [U("Подагра в ремиссии уже год.")],
       [chron("подагра", "Подагра в ремиссии уже год", status="remission")],
       expect=[F(CH, "self", "подагра", "remission")])

single("g-chr-04", "chronic", ["negation"],
       [U("Гастрита и язвы у меня нет.")],
       [chron("гастрит", "Гастрита и язвы у меня нет", pol="negative"),
        chron("язва", "Гастрита и язвы у меня нет", pol="negative")],
       expect=[F(CH, "self", "гастрит", "denied"), F(CH, "self", "язвенная болезнь желудка", "denied")])

single("g-chr-05", "chronic", ["confirmed_absent", "family"],
       [U("Папа у нас здоровый, никаких хронических диагнозов.")],
       statuses=[st(CH, "confirmed_absent", "никаких хронических диагнозов", subj="папа")],
       expect=[], states=[S(CH, "папа", "confirmed_absent")])

single("g-chr-06", "chronic", ["dont_know"],
       [U("Есть ли у меня что-то хроническое — не знаю, давно не обследовалась.")],
       statuses=[st(CH, "dont_know", "Есть ли у меня что-то хроническое — не знаю")],
       expect=[], states=[S(CH, "self", "dont_know")])

single("g-chr-07", "chronic", ["multi_fact", "confirmed_absent"],
       [U("Сахарный диабет 2 типа, на диете, таблеток не пью.")],
       [chron("сахарный диабет 2 типа", "Сахарный диабет 2 типа", status="active")],
       [st(M, "confirmed_absent", "таблеток не пью")],
       expect=[F(CH, "self", "сахарный диабет 2 типа")], states=[S(M, "self", "confirmed_absent")])

# =========================================================================== беременность и лактация

single("g-spec-01", "special_states", ["pregnancy_week"],
       [U("Я беременна, срок 8 недель.")],
       [spec("беременность", "Я беременна, срок 8 недель", pregnancy_week=8)],
       expect=[F(SP, "self", "беременность")])

single("g-spec-02", "special_states", ["lactation"],
       [U("Сейчас кормлю грудью, поэтому нужно что-то безопасное.")],
       [spec("лактация", "Сейчас кормлю грудью")],
       expect=[F(SP, "self", "лактация")])

single("g-spec-03", "special_states", ["planning"],
       [U("Планируем с мужем беременность, это важно учитывать.")],
       [spec("планирование беременности", "Планируем с мужем беременность")],
       expect=[F(SP, "self", "планирование беременности")])

single("g-spec-04", "special_states", ["negation"],
       [U("Нет, я не беременна и не кормлю.")],
       [spec("беременность", "я не беременна", pol="negative"),
        spec("лактация", "и не кормлю", pol="negative")],
       expect=[F(SP, "self", "беременность", "denied"), F(SP, "self", "лактация", "denied")])

single("g-spec-05", "special_states", ["assistant_qa", "negation"],
       [A("Подскажите, вы сейчас не беременны?"), U("Нет.")],
       [spec("беременность", "Нет.", pol="negative")],
       expect=[F(SP, "self", "беременность", "denied")])

single("g-spec-06", "special_states", ["family"],
       [U("Беру для жены: у неё сейчас седьмой месяц, ждём ребёнка.")],
       [spec("беременность", "у неё сейчас седьмой месяц, ждём ребёнка", subj="жена")],
       expect=[F(SP, "жена", "беременность")])

# =========================================================================== возраст, пол, вес

single("g-dem-01", "demographics", ["age"],
       [U("Мне 67 лет.")],
       [dem("age", 67, "Мне 67 лет")],
       expect=[F(DE, "self", "age")])

single("g-dem-02", "demographics", ["child", "weight"],
       [U("Сыну 4 года, вес 17 килограммов.")],
       [dem("age", 4, "Сыну 4 года", "сын"), dem("weight_kg", 17, "вес 17 килограммов", "сын")],
       expect=[F(DE, "сын", "age"), F(DE, "сын", "weight_kg")])

single("g-dem-03", "demographics", ["sex", "weight"],
       [U("Я мужчина, 38 лет, вешу 92 кг.")],
       [dem("sex", "м", "Я мужчина"), dem("age", 38, "38 лет"), dem("weight_kg", 92, "вешу 92 кг")],
       expect=[F(DE, "self", "sex"), F(DE, "self", "age"), F(DE, "self", "weight_kg")])

single("g-dem-04", "demographics", ["family", "age"],
       [U("Бабушке 81 год.")],
       [dem("age", 81, "Бабушке 81 год", "бабушка")],
       expect=[F(DE, "бабушка", "age")])

single("g-dem-05", "demographics", ["child", "weight"],
       [U("Дочке 12, весит около 40 кг.")],
       [dem("age", 12, "Дочке 12", "дочь"), dem("weight_kg", 40, "весит около 40 кг", "дочь")],
       expect=[F(DE, "дочь", "age"), F(DE, "дочь", "weight_kg")])

# =========================================================================== образ жизни

single("g-life-01", "lifestyle", ["negation"],
       [U("Не курю, алкоголь не пью вообще.")],
       [life("smoking", "нет", "Не курю"), life("alcohol", "нет", "алкоголь не пью вообще")],
       expect=[F(LI, "self", "smoking"), F(LI, "self", "alcohol")])

single("g-life-02", "lifestyle", ["details"],
       [U("Курю, но мало — штук пять в день.")],
       [life("smoking", "да", "Курю, но мало — штук пять в день", details="штук пять в день")],
       expect=[F(LI, "self", "smoking")])

single("g-life-03", "lifestyle", ["implicit"],
       [U("Я водитель автобуса.")],
       [life("driving", "да", "Я водитель автобуса", details="водитель автобуса")],
       expect=[F(LI, "self", "driving")])

single("g-life-04", "lifestyle", ["multi_fact"],
       [U("Выпиваю иногда бокал вина по праздникам, за руль не сажусь.")],
       [life("alcohol", "иногда", "Выпиваю иногда бокал вина по праздникам", details="бокал вина по праздникам"),
        life("driving", "нет", "за руль не сажусь")],
       expect=[F(LI, "self", "alcohol"), F(LI, "self", "driving")])

# =========================================================================== симптомы

single("g-sym-01", "symptoms", ["negation"],
       [U("Болит горло, глотать больно, температуры нет.")],
       [sym("болит горло", "Болит горло, глотать больно", status="active"),
        sym("температура", "температуры нет", pol="negative")],
       expect=[F(SY, "self", "боль в горле"), F(SY, "self", "повышенная температура", "denied")])

single("g-sym-02", "symptoms", ["duration"],
       [U("Уже неделю сухой кашель.")],
       [sym("сухой кашель", "Уже неделю сухой кашель", status="active", duration="неделю")],
       expect=[F(SY, "self", "кашель")])

single("g-sym-03", "symptoms", ["multi_fact"],
       [U("Тошнит с утра и понос.")],
       [sym("тошнота", "Тошнит с утра", status="active"), sym("понос", "и понос", status="active")],
       expect=[F(SY, "self", "тошнота"), F(SY, "self", "диарея")])

single("g-sym-04", "symptoms", ["severity"],
       [U("Раскалывается голова, очень сильно.")],
       [sym("раскалывается голова", "Раскалывается голова, очень сильно", status="active", severity="сильная")],
       expect=[F(SY, "self", "головная боль")])

single("g-sym-05", "symptoms", ["resolved"],
       [U("Изжога прошла, а запор остался.")],
       [sym("изжога", "Изжога прошла", status="resolved"), sym("запор", "а запор остался", status="active")],
       expect=[F(SY, "self", "изжога", "resolved"), F(SY, "self", "запор")])

single("g-sym-06", "symptoms", ["simple"],
       [U("Болит спина после дачи.")],
       [sym("болит спина", "Болит спина после дачи", status="active")],
       expect=[F(SY, "self", "боль в спине")])

single("g-sym-07", "symptoms", ["family", "duration"],
       [U("У дочки зуб болит второй день.")],
       [sym("зуб болит", "У дочки зуб болит второй день", subj="дочь", status="active", duration="второй день")],
       expect=[F(SY, "дочь", "зубная боль")])

# =========================================================================== диагнозы

single("g-dia-01", "diagnoses", ["by_doctor"],
       [U("Врач сказал — ангина.")],
       [diag("ангина", "Врач сказал — ангина", status="active", diagnosed_by="врач")],
       expect=[F(DI, "self", "ангина")])

single("g-dia-02", "diagnoses", ["self_diagnosed"],
       [U("Думаю, у меня обычная простуда.")],
       [diag("простуда", "Думаю, у меня обычная простуда", status="active", diagnosed_by="сам")],
       expect=[F(DI, "self", "ОРВИ")])

single("g-dia-03", "diagnoses", ["resolved"],
       [U("Гайморит вылечили, всё прошло.")],
       [diag("гайморит", "Гайморит вылечили, всё прошло", status="resolved")],
       expect=[F(DI, "self", "синусит", "resolved")])

single("g-dia-04", "diagnoses", ["family", "by_doctor"],
       [U("У мужа грипп, подтвердили в поликлинике.")],
       [diag("грипп", "У мужа грипп, подтвердили в поликлинике", subj="муж", status="active", diagnosed_by="врач")],
       expect=[F(DI, "муж", "грипп")])

# =========================================================================== короткие ответы на вопросы ассистента

single("g-qa-01", "assistant_qa", ["confirmed_absent"],
       [A("Вы принимаете какие-нибудь лекарства постоянно?"), U("Нет.")],
       statuses=[st(M, "confirmed_absent", "Нет.")],
       expect=[], states=[S(M, "self", "confirmed_absent")])

single("g-qa-02", "assistant_qa", ["dont_know"],
       [A("Есть ли у вас аллергия на какие-то препараты?"), U("Не знаю.")],
       statuses=[st(AL, "dont_know", "Не знаю.")],
       expect=[], states=[S(AL, "self", "dont_know")])

single("g-qa-03", "assistant_qa", ["child", "number_as_word"],
       [A("Сколько лет ребёнку?"), U("Семь.")],
       [dem("age", 7, "Семь.", "ребёнок")],
       expect=[F(DE, "ребёнок", "age")],
       note="возраст словом: модель должна вернуть число")

single("g-qa-04", "assistant_qa", ["chronic"],
       [A("Есть ли хронические заболевания — давление, диабет?"), U("Давление бывает повышенное.")],
       [chron("давление", "Давление бывает повышенное", status="active")],
       expect=[F(CH, "self", "артериальная гипертензия")])

single("g-qa-05", "assistant_qa", ["lactation"],
       [A("Вы беременны или кормите грудью?"), U("Кормлю.")],
       [spec("лактация", "Кормлю.")],
       expect=[F(SP, "self", "лактация")])

single("g-qa-06", "assistant_qa", ["family", "multi_fact"],
       [U("Муж кашляет вторую неделю."), A("Он курит?"), U("Да, много.")],
       [sym("кашель", "Муж кашляет вторую неделю", subj="муж", status="active", duration="вторую неделю"),
        life("smoking", "да", "Да, много.", subj="муж", details="много")],
       expect=[F(SY, "муж", "кашель"), F(LI, "муж", "smoking")])

# =========================================================================== ловушки

single("g-trap-01", "trap", ["trap_question"],
       [U("Можно ли пить Ибуклин вместе с Терафлю?")],
       expect=[], note="препараты только в вопросе — фактов нет")

single("g-trap-02", "trap", ["trap_assistant_drug", "confirmed_absent"],
       [U("Мучает изжога, что посоветуете?"),
        A("Можно попробовать Омез или Нольпазу, курсом до двух недель. Вы принимаете другие лекарства?"),
        U("Ничего другого не пью.")],
       [sym("изжога", "Мучает изжога", status="active")],
       [st(M, "confirmed_absent", "Ничего другого не пью.")],
       expect=[F(SY, "self", "изжога")], states=[S(M, "self", "confirmed_absent")],
       note="препараты назвал ассистент, а не покупатель")

single("g-trap-03", "trap", ["trap_hypothetical", "trap_question"],
       [U("А беременным Но-шпу можно? Спрашиваю для себя на будущее.")],
       expect=[], note="гипотетический вопрос — беременности нет")

single("g-trap-04", "trap", ["resolved_first"],
       [U("Голова болела вчера, сегодня всё нормально.")],
       [sym("головная боль", "Голова болела вчера, сегодня всё нормально", status="resolved")],
       expect=[F(SY, "self", "головная боль", "resolved")])

single("g-trap-05", "trap", ["trap_generic"],
       [U("Посоветуйте что-нибудь от аллергии.")],
       expect=[], note="аллерген не назван — факта раздела allergies нет")

single("g-trap-06", "trap", ["trap_question", "family"],
       [U("Мама просила узнать, можно ли ей Найз — у неё гастрит.")],
       [chron("гастрит", "у неё гастрит", subj="мама", status="active")],
       expect=[F(CH, "мама", "гастрит")])

single("g-trap-07", "trap", ["trap_assistant_drug", "family"],
       [U("У сына температура 38,5, чем сбить?"),
        A("Подойдут Нурофен или Панадол в детской форме, доза зависит от веса. Сколько весит сын?"),
        U("25 кг.")],
       [sym("температура", "У сына температура 38,5", subj="сын", status="active"),
        dem("weight_kg", 25, "25 кг.", "сын")],
       expect=[F(SY, "сын", "повышенная температура"), F(DE, "сын", "weight_kg")])

single("g-trap-08", "trap", ["trap_third_party"],
       [U("Спрашиваю не для себя, а для соседа: можно ли Цетрин с алкоголем?")],
       expect=[], note="вопрос за третье лицо, о нём ничего не сообщено")

# =========================================================================== несколько разделов в одной реплике

single("g-multi-01", "multi_fact", ["multi_section"],
       [U("Мне 45, гипертония, пью Престариум, аллергия на пенициллин.")],
       [dem("age", 45, "Мне 45"), chron("гипертония", "гипертония", status="active"),
        med("Престариум", "пью Престариум", status="active"),
        alg("пенициллин", "аллергия на пенициллин", reaction_type="allergy")],
       expect=[F(DE, "self", "age"), F(CH, "self", "артериальная гипертензия"), F(M, "self", "периндоприл"),
               F(AL, "self", "пенициллины")])

single("g-multi-02", "multi_fact", ["multi_section", "child", "confirmed_absent"],
       [U("Дочке 9 лет, 30 кг. Аллергий у неё не было. Кашель с понедельника.")],
       [dem("age", 9, "Дочке 9 лет", "дочь"), dem("weight_kg", 30, "30 кг", "дочь"),
        sym("кашель", "Кашель с понедельника", subj="дочь", status="active", duration="с понедельника")],
       [st(AL, "confirmed_absent", "Аллергий у неё не было", subj="дочь")],
       expect=[F(DE, "дочь", "age"), F(DE, "дочь", "weight_kg"), F(SY, "дочь", "кашель")],
       states=[S(AL, "дочь", "confirmed_absent")])

single("g-multi-03", "multi_fact", ["multi_section", "confirmed_absent", "pregnancy_week"],
       [U("Беременна, 20 недель, изжога замучила, лекарства никакие не пью.")],
       [spec("беременность", "Беременна, 20 недель", pregnancy_week=20),
        sym("изжога", "изжога замучила", status="active")],
       [st(M, "confirmed_absent", "лекарства никакие не пью")],
       expect=[F(SP, "self", "беременность"), F(SY, "self", "изжога")], states=[S(M, "self", "confirmed_absent")])

single("g-multi-04", "multi_fact", ["multi_section", "family"],
       [U("У папы диабет и давление, пьёт Сиофор и Диротон, на аспирин аллергия.")],
       [chron("диабет", "У папы диабет", subj="папа", status="active"),
        chron("давление", "и давление", subj="папа", status="active"),
        med("Сиофор", "пьёт Сиофор", subj="папа", status="active"),
        med("Диротон", "и Диротон", subj="папа", status="active"),
        alg("аспирин", "на аспирин аллергия", subj="папа", reaction_type="allergy")],
       expect=[F(CH, "папа", "сахарный диабет"), F(CH, "папа", "артериальная гипертензия"),
               F(M, "папа", "метформин"), F(M, "папа", "лизиноприл"),
               F(AL, "папа", "ацетилсалициловая кислота")])

single("g-multi-05", "multi_fact", ["multi_section"],
       [U("Курю, за рулём каждый день, Тавегил пью от аллергии на пыльцу.")],
       [life("smoking", "да", "Курю"), life("driving", "да", "за рулём каждый день", details="каждый день"),
        med("Тавегил", "Тавегил пью от аллергии на пыльцу", status="active", indication="аллергия на пыльцу"),
        alg("пыльца", "от аллергии на пыльцу", reaction_type="allergy")],
       expect=[F(LI, "self", "smoking"), F(LI, "self", "driving"), F(M, "self", "клемастин"),
               F(AL, "self", "пыльца растений")])

single("g-multi-06", "multi_fact", ["multi_section", "multi_drug"],
       [U("Мне 72, ИБС, после стентирования пью Плавикс и Аторис.")],
       [dem("age", 72, "Мне 72"), chron("ИБС", "ИБС", status="active"),
        med("Плавикс", "пью Плавикс", status="active"), med("Аторис", "и Аторис", status="active")],
       expect=[F(DE, "self", "age"), F(CH, "self", "ишемическая болезнь сердца"), F(M, "self", "клопидогрел"),
               F(M, "self", "аторвастатин")])

single("g-multi-07", "multi_fact", ["multi_section", "negation", "self_diagnosed"],
       [U("Простуда, температура 37,5, заложен нос, кашля нет.")],
       [diag("простуда", "Простуда", status="active", diagnosed_by="сам"),
        sym("температура", "температура 37,5", status="active"), sym("заложен нос", "заложен нос", status="active"),
        sym("кашель", "кашля нет", pol="negative")],
       expect=[F(DI, "self", "ОРВИ"), F(SY, "self", "повышенная температура"), F(SY, "self", "насморк"),
               F(SY, "self", "кашель", "denied")])

single("g-multi-08", "multi_fact", ["multi_section", "family", "intolerance", "remission"],
       [U("Мужу 50, у него язва в ремиссии, и он плохо переносит Диклофенак.")],
       [dem("age", 50, "Мужу 50", "муж"), chron("язва", "у него язва в ремиссии", subj="муж", status="remission"),
        alg("Диклофенак", "он плохо переносит Диклофенак", subj="муж", reaction_type="intolerance")],
       expect=[F(DE, "муж", "age"), F(CH, "муж", "язвенная болезнь желудка", "remission"),
               F(AL, "муж", "диклофенак")])

# =========================================================================== разные субъекты в одном диалоге

single("g-subj-01", "subjects", ["mixed_subjects"],
       [U("У меня аллергия на мёд, а у дочки — на цитрусовые.")],
       [alg("мёд", "У меня аллергия на мёд", reaction_type="allergy"),
        alg("цитрусовые", "а у дочки — на цитрусовые", subj="дочь", reaction_type="allergy")],
       expect=[F(AL, "self", "мёд"), F(AL, "дочь", "цитрусовые")])

single("g-subj-02", "subjects", ["mixed_subjects"],
       [U("Мне 30, а маме 58, у неё давление.")],
       [dem("age", 30, "Мне 30"), dem("age", 58, "а маме 58", "мама"),
        chron("давление", "у неё давление", subj="мама", status="active")],
       expect=[F(DE, "self", "age"), F(DE, "мама", "age"), F(CH, "мама", "артериальная гипертензия")])

single("g-subj-03", "subjects", ["mixed_subjects", "negation"],
       [U("Я не курю, а муж курит.")],
       [life("smoking", "нет", "Я не курю"), life("smoking", "да", "а муж курит", subj="муж")],
       expect=[F(LI, "self", "smoking"), F(LI, "муж", "smoking")])

# =========================================================================== изменения уже сохранённого

single("g-chg-01", "changes", ["change", "dosage"],
       [U("Конкор теперь пью 5 мг вместо 2,5.")],
       [med("Конкор", "Конкор теперь пью 5 мг вместо 2,5", status="active", dosage="5 мг")],
       expect=[F(M, "self", "бисопролол")], data=[[M, "self", "бисопролол", "dosage", "5 мг"]],
       setup={"facts": [med("Конкор 2,5 мг", "пью Конкор 2,5 мг", status="active")]})

single("g-chg-02", "changes", ["confirmation"],
       [U("Аллергия на пенициллин подтвердилась, проверяли у аллерголога.")],
       [alg("пенициллин", "Аллергия на пенициллин подтвердилась", reaction_type="allergy")],
       expect=[F(AL, "self", "пенициллины")],
       setup={"facts": [alg("пенициллин", "аллергия на пенициллин")]})

single("g-chg-03", "changes", ["confirmation"],
       [U("Кашель стал влажным, но не проходит.")],
       [sym("кашель", "Кашель стал влажным, но не проходит", status="active")],
       expect=[F(SY, "self", "кашель")],
       setup={"facts": [sym("кашель", "кашель", status="active")]})

single("g-chg-04", "changes", ["change", "implicit_negation", "lactation"],
       [U("Родила месяц назад, теперь кормлю.")],
       [spec("беременность", "Родила месяц назад", pol="negative"), spec("лактация", "теперь кормлю")],
       expect=[F(SP, "self", "беременность", "denied"), F(SP, "self", "лактация")],
       setup={"facts": [spec("беременность", "беременна, 12 недель", pregnancy_week=12)]},
       note="«родила» — беременности больше нет")

single("g-chg-05", "changes", ["state_cleared"],
       [U("Начала пить Эриус от аллергии.")],
       [med("Эриус", "Начала пить Эриус от аллергии", status="active", indication="аллергия")],
       expect=[F(M, "self", "дезлоратадин")], states=[],
       setup={"facts": [], "section_statuses": [st(M, "confirmed_absent", "ничего не принимаю")]})

single("g-chg-06", "changes", ["reactivation"],
       [U("После операции снова пью Ксарелто.")],
       [med("Ксарелто", "После операции снова пью Ксарелто", status="active")],
       expect=[F(M, "self", "ривароксабан")],
       setup={"facts": [med("Ксарелто", "Ксарелто больше не пью", status="stopped")]})

# =========================================================================== многосессионные сценарии (E2E)

PRESSURE = "давлен|гипертон|лозап|лозартан"

scenario("g-e2e-01", "multi_session", ["e2e", "change", "history"], [
    session([U("Здравствуйте. Давление повышенное уже много лет, по утрам Энап 10."),
             A("Есть ли у вас аллергия на лекарства?"), U("Нет.")],
            [chron("давление повышенное", "Давление повышенное уже много лет", status="active"),
             med("Энап 10", "по утрам Энап 10", status="active", frequency="по утрам")],
            [st(AL, "confirmed_absent", "Нет.")],
            expect=[F(CH, "self", "артериальная гипертензия"), F(M, "self", "эналаприл")],
            states=[S(AL, "self", "confirmed_absent")]),
    session([U("С прошлой недели вместо Энапа пью Лозап 50, так решил кардиолог.")],
            [med("Энап", "вместо Энапа", status="stopped"), med("Лозап 50", "пью Лозап 50", status="active")],
            expect=[F(CH, "self", "артериальная гипертензия"), F(M, "self", "эналаприл", "stopped"),
                    F(M, "self", "лозартан")],
            states=[S(AL, "self", "confirmed_absent")], after_days=30),
    probe("Разболелась голова — что можно принять?", ["symptom_relief"],
          include=[(M, "лозартан"), (CH, "артериальная гипертензия")], exclude=[(M, "эналаприл")],
          clarify=[(SP, "unknown")], no_clarify=[AL, M, CH],
          checks=[mentions(PRESSURE, "учтена гипертония и текущий препарат")], after_days=1),
    probe("Какие таблетки от давления у меня были до нынешних?", ["therapy_history"],
          include=[(M, "эналаприл"), (M, "лозартан")],
          checks=[mentions("энап|эналаприл", "назван прежний препарат")]),
])

scenario("g-e2e-02", "multi_session", ["e2e", "child", "dosing"], [
    session([U("Сыну 6 лет, весит 22 кг. Аллергия у него на Амоксиклав — сыпь была.")],
            [dem("age", 6, "Сыну 6 лет", "сын"), dem("weight_kg", 22, "весит 22 кг", "сын"),
             alg("Амоксиклав", "Аллергия у него на Амоксиклав — сыпь была", subj="сын", reaction_type="allergy",
                 reaction="сыпь")],
            expect=[F(DE, "сын", "age"), F(DE, "сын", "weight_kg"),
                    F(AL, "сын", "амоксициллин + клавулановая кислота")]),
    probe("Сколько Нурофена можно дать сыну?", ["dosing"], subject="сын",
          include=[(DE, "age"), (DE, "weight_kg")], no_clarify=[SP, DE],
          checks=[not_mentions(r"\d+([.,]\d+)?\s*(мг|мл)\b", "доза не названа — только инструкция"),
                  mentions("инструкц|фармацевт|врач|педиатр", "отсылка к инструкции или специалисту")],
          after_days=10),
    probe("Что дать сыну от температуры?", ["symptom_relief"], subject="сын",
          include=[(AL, "амоксициллин + клавулановая кислота")],
          clarify=[(M, "unknown"), (CH, "unknown")], no_clarify=[AL, SP],
          checks=[mentions("лекарств|принима|хронич|заболеван", "уточняет недостающее")]),
])

scenario("g-e2e-03", "multi_session", ["e2e", "pregnancy", "lactation", "change"], [
    session([U("Мне 31, жду ребёнка — сейчас 10 недель.")],
            [spec("беременность", "жду ребёнка — сейчас 10 недель", pregnancy_week=10), dem("age", 31, "Мне 31")],
            expect=[F(SP, "self", "беременность"), F(DE, "self", "age")]),
    probe("Что можно от насморка?", ["symptom_relief"], include=[(SP, "беременность")], no_clarify=[SP],
          checks=[mentions("беремен", "учтена беременность")], after_days=1),
    session([U("Я родила, сейчас кормлю грудью.")],
            [spec("беременность", "Я родила", pol="negative"), spec("лактация", "сейчас кормлю грудью")],
            expect=[F(SP, "self", "беременность", "denied"), F(SP, "self", "лактация"), F(DE, "self", "age")],
            after_days=240),
    probe("Посоветуйте что-нибудь от головной боли.", ["symptom_relief"], include=[(SP, "лактация")],
          no_clarify=[SP], checks=[mentions("корм|грудн|лактац", "учтено кормление грудью")], after_days=1),
])

scenario("g-e2e-04", "multi_session", ["e2e", "state_cleared", "drug_allergy"], [
    session([A("Есть ли у вас аллергия на лекарства?"), U("Нет, никогда не было.")],
            statuses=[st(AL, "confirmed_absent", "Нет, никогда не было.")],
            expect=[], states=[S(AL, "self", "confirmed_absent")]),
    session([U("Оказалось, на Флемоксин у меня аллергия — после него пошла сыпь.")],
            [alg("Флемоксин", "на Флемоксин у меня аллергия — после него пошла сыпь", reaction_type="allergy",
                 reaction="сыпь")],
            expect=[F(AL, "self", "амоксициллин")], states=[], after_days=20),
    probe("Чем заменить Флемоксин?", ["analog"], include=[(AL, "амоксициллин")], no_clarify=[AL],
          checks=[mentions("аллерг", "учтена аллергия")], after_days=1),
])

scenario("g-e2e-05", "multi_session", ["e2e", "correction", "negation"], [
    session([U("У меня аллергия на Аспирин.")],
            [alg("Аспирин", "У меня аллергия на Аспирин", reaction_type="allergy")],
            expect=[F(AL, "self", "ацетилсалициловая кислота")]),
    session([U("Насчёт аспирина я ошиблась — это была не аллергия, аллергии на него нет.")],
            [alg("аспирин", "аллергии на него нет", pol="negative")],
            expect=[F(AL, "self", "ацетилсалициловая кислота", "denied")], after_days=5),
    probe("Что-нибудь от зубной боли посоветуете?", ["symptom_relief"],
          include=[(AL, "ацетилсалициловая кислота")], clarify=[(AL, "unknown")], after_days=1),
])

scenario("g-e2e-06", "multi_session", ["e2e", "ttl", "forgetting"], [
    session([U("Третий день кашель и болит горло.")],
            [sym("кашель", "Третий день кашель", status="active", duration="третий день"),
             sym("болит горло", "болит горло", status="active")],
            expect=[F(SY, "self", "кашель"), F(SY, "self", "боль в горле")]),
    probe("Что выпить от кашля?", ["symptom_relief"], include=[(SY, "кашель"), (SY, "боль в горле")],
          after_days=1),
    probe("Посоветуйте что-нибудь от изжоги.", ["symptom_relief"],
          exclude=[(SY, "кашель"), (SY, "боль в горле")], after_days=20),
    probe("Расскажи, какие сведения обо мне ты хранишь.", ["profile_review"], include=[(SY, "кашель")]),
])

scenario("g-e2e-07", "multi_session", ["e2e", "conflict"], [
    session([U("На Сумамед у меня аллергия, отёк был.")],
            [alg("Сумамед", "На Сумамед у меня аллергия, отёк был", reaction_type="allergy", reaction="отёк")],
            expect=[F(AL, "self", "азитромицин")]),
    session([A("Подскажите, аллергия на какие-то препараты есть?"), U("Нет, ни на что.")],
            statuses=[st(AL, "confirmed_absent", "Нет, ни на что.")],
            expect=[F(AL, "self", "азитромицин")], states=[], after_days=60),
    probe("Что можно от простуды?", ["symptom_relief"], include=[(AL, "азитромицин")],
          clarify=[(AL, "conflict")], checks=[mentions("аллерг|сумамед|азитромицин", "противоречие не скрыто")],
          after_days=1),
])

scenario("g-e2e-08", "multi_session", ["e2e", "family", "subject_isolation"], [
    session([U("Покупаю для мамы: ей 74, пьёт Варфарин, и у неё фибрилляция предсердий.")],
            [dem("age", 74, "ей 74", "мама"), med("Варфарин", "пьёт Варфарин", subj="мама", status="active"),
             chron("фибрилляция предсердий", "у неё фибрилляция предсердий", subj="мама", status="active")],
            expect=[F(DE, "мама", "age"), F(M, "мама", "варфарин"), F(CH, "мама", "фибрилляция предсердий")]),
    session([U("А у меня самой гастрит.")],
            [chron("гастрит", "А у меня самой гастрит", status="active")],
            expect=[F(DE, "мама", "age"), F(M, "мама", "варфарин"), F(CH, "мама", "фибрилляция предсердий"),
                    F(CH, "self", "гастрит")], after_days=3),
    probe("Подойдёт ли маме Найз с учётом её лечения?", ["drug_compatibility"], subject="мама",
          include=[(M, "варфарин"), (CH, "фибрилляция предсердий")], exclude=[(CH, "гастрит")],
          no_clarify=[M, SP], checks=[mentions("варфарин|антикоагул|кровотеч", "учтён варфарин")], after_days=1),
    probe("А мне что можно от изжоги?", ["symptom_relief"],
          include=[(CH, "гастрит")], exclude=[(M, "варфарин")], clarify=[(M, "unknown")]),
])

scenario("g-e2e-09", "multi_session", ["e2e", "lifestyle"], [
    session([U("Я мужчина, 40 лет, таксист — весь день за рулём. Аллергий и хронических болезней нет, "
               "лекарств не принимаю.")],
            [dem("sex", "м", "Я мужчина"), dem("age", 40, "40 лет"),
             life("driving", "да", "таксист — весь день за рулём", details="таксист, весь день за рулём")],
            [st(AL, "confirmed_absent", "Аллергий и хронических болезней нет"),
             st(CH, "confirmed_absent", "Аллергий и хронических болезней нет"),
             st(M, "confirmed_absent", "лекарств не принимаю")],
            expect=[F(DE, "self", "sex"), F(DE, "self", "age"), F(LI, "self", "driving")],
            states=[S(AL, "self", "confirmed_absent"), S(CH, "self", "confirmed_absent"),
                    S(M, "self", "confirmed_absent")]),
    probe("Посоветуйте что-нибудь от аллергии на пыльцу.", ["symptom_relief"], include=[(LI, "driving")],
          no_clarify=[AL, M, CH, SP],
          checks=[mentions("сонлив|за рул|вожд|водит|автомоб|машин|концентрац|внимани", "учтено вождение")],
          after_days=2),
])

scenario("g-e2e-10", "multi_session", ["e2e", "change", "dosage"], [
    session([U("Пью Глюкофаж 500 два раза в день, у меня диабет 2 типа.")],
            [med("Глюкофаж 500", "Пью Глюкофаж 500 два раза в день", status="active", frequency="два раза в день"),
             chron("диабет 2 типа", "у меня диабет 2 типа", status="active")],
            expect=[F(M, "self", "метформин"), F(CH, "self", "сахарный диабет 2 типа")]),
    session([U("Врач увеличил Глюкофаж до 1000 мг.")],
            [med("Глюкофаж", "Врач увеличил Глюкофаж до 1000 мг", status="active", dosage="1000 мг")],
            expect=[F(M, "self", "метформин"), F(CH, "self", "сахарный диабет 2 типа")],
            data=[[M, "self", "метформин", "dosage", "1000 мг"]], after_days=45),
    probe("Можно ли мне с моими таблетками Нурофен?", ["drug_compatibility"],
          include=[(M, "метформин"), (CH, "сахарный диабет 2 типа")],
          checks=[mentions("метформин|глюкофаж|диабет", "учтена терапия диабета")], after_days=1),
])

scenario("g-e2e-11", "multi_session", ["e2e", "state_cleared"], [
    session([A("Вы принимаете что-то постоянно?"), U("Ничего не принимаю.")],
            statuses=[st(M, "confirmed_absent", "Ничего не принимаю.")],
            expect=[], states=[S(M, "self", "confirmed_absent")]),
    probe("Посоветуйте что-нибудь от насморка.", ["symptom_relief"], no_clarify=[M], after_days=1),
    session([U("Теперь пью Аторис, врач назначил от холестерина, уже неделю.")],
            [med("Аторис", "Теперь пью Аторис, врач назначил от холестерина, уже неделю", status="active",
                 indication="холестерин")],
            expect=[F(M, "self", "аторвастатин")], states=[], after_days=14),
    probe("А от насморка теперь что можно?", ["symptom_relief"], include=[(M, "аторвастатин")], no_clarify=[M],
          after_days=1),
])

scenario("g-e2e-12", "multi_session", ["e2e", "reactivation", "history"], [
    session([U("Раньше пил Омез, сейчас не пью.")],
            [med("Омез", "Раньше пил Омез, сейчас не пью", status="stopped")],
            expect=[F(M, "self", "омепразол", "stopped")]),
    probe("Что я принимал раньше?", ["therapy_history"], include=[(M, "омепразол")], after_days=1),
    session([U("Опять каждое утро Омез — изжога вернулась.")],
            [med("Омез", "Опять каждое утро Омез", status="active", frequency="каждое утро"),
             sym("изжога", "изжога вернулась", status="active")],
            expect=[F(M, "self", "омепразол"), F(SY, "self", "изжога")], after_days=30),
    probe("Можно ли мне Смекту с моими лекарствами?", ["drug_compatibility"], include=[(M, "омепразол")],
          no_clarify=[M], after_days=1),
])

# =========================================================================== запросы

QUERIES = [
    ("Что принять от боли в животе?", ["symptom_relief"], "self"),
    ("Голова раскалывается, что выпить?", ["symptom_relief"], "self"),
    ("Можно ли совмещать Эналаприл и Ибупрофен?", ["drug_compatibility"], "self"),
    ("Совместим ли Варфарин с Аспирином?", ["drug_compatibility"], "self"),
    ("Можно ли выпить вина, если я пью Метронидазол?", ["drug_compatibility"], "self"),
    ("Как правильно пить Смекту?", ["dosing"], "self"),
    ("Сколько раз в день капать Називин?", ["dosing"], "self"),
    ("Какая дозировка Парацетамола для ребёнка 5 лет?", ["dosing"], "ребёнок"),
    ("Есть что-то дешевле Нурофена?", ["analog"], "self"),
    ("Чем можно заменить Эриус?", ["analog"], "self"),
    ("Посоветуйте аналог Ксарелто подешевле", ["analog"], "self"),
    ("Можно ли Цетрин при грудном вскармливании?", ["pregnancy_lactation"], "self"),
    ("Я беременна, что можно от изжоги?", ["pregnancy_lactation", "symptom_relief"], "self"),
    ("Жена беременна, ей можно Парацетамол?", ["pregnancy_lactation"], "жена"),
    ("Какие таблетки я брал в прошлый раз?", ["therapy_history"], "self"),
    ("Напомни, что мне назначали от давления раньше", ["therapy_history"], "self"),
    ("Что у тебя записано про мою аллергию?", ["profile_review"], "self"),
    ("Покажи, что ты обо мне знаешь", ["profile_review"], "self"),
    ("У вас есть Кагоцел в наличии?", ["general"], "self"),
    ("Вы работаете в воскресенье?", ["general"], "self"),
    ("Можно оплатить картой при доставке?", ["general"], "self"),
    ("Здравствуйте! Я кормлю грудью и пью Эутирокс.", ["general"], "self"),
    ("Чем маме сбить жар под 39?", ["symptom_relief"], "мама"),
    ("Чем лечить насморк у ребёнка?", ["symptom_relief"], "ребёнок"),
    ("Сын поцарапался, чем обработать?", ["symptom_relief"], "сын"),
    ("Папе можно Кеторол при язве?", ["drug_compatibility"], "папа"),
    ("Можно ли Супрастин и Тавегил вместе?", ["drug_compatibility"], "self"),
    ("Сколько дней можно пить Амоксиклав?", ["dosing"], "self"),
    ("Как давать Смекту годовалому малышу?", ["dosing"], "ребёнок"),
    ("Есть ли замена Конкору без рецепта?", ["analog"], "self"),
    ("Можно ли Но-шпу при кормлении?", ["pregnancy_lactation"], "self"),
    ("Что мне раньше помогало от изжоги?", ["therapy_history"], "self"),
    ("Что посоветуете от бессонницы?", ["symptom_relief"], "self"),
    ("Нужно что-то от укачивания для дочки в дорогу", ["symptom_relief"], "дочь"),
    ("Можно ли пить Найз и Кардиомагнил одновременно?", ["drug_compatibility"], "self"),
    ("Какой аналог у Омеза?", ["analog"], "self"),
    ("Есть ли у вас тесты на беременность?", ["general"], "self"),
    ("Мне при диабете можно сироп от кашля с сахаром?", ["drug_compatibility"], "self"),
    ("Бабушке плохо, давление поднялось до 180, что дать?", ["symptom_relief"], "бабушка"),
    ("Сколько таблеток Нурофена можно выпить за сутки?", ["dosing"], "self"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "scenarios.jsonl", "w", encoding="utf-8") as fh:
        for sc in SCENARIOS:
            fh.write(json.dumps(sc, ensure_ascii=False) + "\n")
    with open(OUT / "queries.jsonl", "w", encoding="utf-8") as fh:
        for i, (text, types, subject) in enumerate(QUERIES, 1):
            fh.write(json.dumps({"id": f"gq{i:02d}", "text": text, "query_types": types, "subject": subject},
                                ensure_ascii=False) + "\n")
    sessions = sum(1 for sc in SCENARIOS for s in sc["steps"] if s["type"] == "session")
    probes = sum(1 for sc in SCENARIOS for s in sc["steps"] if s["type"] == "probe")
    multi = sum(1 for sc in SCENARIOS if sc["category"] == "multi_session")
    print(f"сценариев: {len(SCENARIOS)} (многосессионных: {multi}), сессий: {sessions}, проб: {probes}, "
          f"запросов: {len(QUERIES)}")


if __name__ == "__main__":
    main()
