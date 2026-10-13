"""Промпты LLM-шагов.

Принцип из статьи: extractor получает правила только выбранного раздела.
Few-shot примеры хранятся как Python-структуры и сериализуются в JSON,
поэтому они всегда синтаксически корректны; тесты дополнительно
прогоняют их через ту же валидацию, что и ответы модели.
"""
from __future__ import annotations

import json
from typing import Any

from .llm import Messages
from .schemas import QueryType
from .sections import POLICIES, Section

ROLE_LABELS = {"user": "Покупатель", "assistant": "Ассистент"}


def format_dialog(turns: list[dict[str, str]]) -> str:
    return "\n".join(f"{ROLE_LABELS.get(t['role'], t['role'])}: {t['content'].strip()}" for t in turns)


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _fewshot(system: str, examples: list[tuple[str, dict]], query: str) -> Messages:
    msgs: Messages = [{"role": "system", "content": system}]
    for user, answer in examples:
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": _j(answer)})
    msgs.append({"role": "user", "content": query})
    return msgs


# =========================================================================== 1. классификатор разделов

SECTION_DESCRIPTIONS: dict[Section, str] = {
    Section.MEDICATIONS: "лекарства, которые человек принимает, принимал, перестал принимать или ему назначили; "
                         "также «никаких лекарств не принимаю»",
    Section.ALLERGIES: "аллергия и непереносимость (лекарства, продукты, пыльца); также «аллергий нет»",
    Section.CHRONIC_CONDITIONS: "хронические заболевания: гипертония, диабет, астма, язва и т. п.; "
                                "также «хронических болезней нет»",
    Section.SPECIAL_STATES: "беременность, кормление грудью, планирование беременности; также «не беременна»",
    Section.DEMOGRAPHICS: "возраст, пол, вес",
    Section.LIFESTYLE: "курение, алкоголь, вождение автомобиля",
    Section.SYMPTOMS: "текущие жалобы и симптомы: боль, температура, кашель, насморк, изжога; также «симптом прошёл»",
    Section.REPORTED_DIAGNOSES: "острые диагнозы, которые назвал врач или сам человек: ОРВИ, цистит, отит",
}

CLASSIFIER_SYSTEM = """Ты — модуль разметки диалогов аптечного чата. Определи, в каких разделах памяти покупатель сообщил сведения о себе или о человеке, для которого покупает.

Разделы:
{sections}

Правила:
- Учитывай только то, что сообщил покупатель. Реплики ассистента — контекст: ответ «нет» на вопрос ассистента об аллергии относится к allergies.
- Вопрос покупателя не является сведением о нём. «Можно ли мне Нурофен?» не значит, что он принимает Нурофен.
- Если сведений нет, верни пустой список.

Ответ — только JSON без пояснений: {{"sections": [...]}}"""

CLASSIFIER_EXAMPLES: list[tuple[list[dict[str, str]], list[str]]] = [
    (
        [{"role": "user", "content": "У меня гипертония, пью Конкор. Подскажите, можно ли с ним Нурофен?"}],
        ["chronic_conditions", "medications"],
    ),
    (
        [
            {"role": "user", "content": "Нужно что-нибудь от кашля для сына."},
            {"role": "assistant", "content": "Подскажу. Сколько сыну лет и есть ли у него аллергия на лекарства?"},
            {"role": "user", "content": "Ему 5 лет, аллергии нет. Кашляет третий день."},
        ],
        ["demographics", "allergies", "symptoms"],
    ),
    ([{"role": "user", "content": "До скольки вы сегодня работаете?"}], []),
]


def section_classifier_messages(turns: list[dict[str, str]]) -> Messages:
    sections = "\n".join(f"- {s.value} — {d}" for s, d in SECTION_DESCRIPTIONS.items())
    system = CLASSIFIER_SYSTEM.format(sections=sections)
    examples = [(f"Диалог:\n{format_dialog(t)}", {"sections": s}) for t, s in CLASSIFIER_EXAMPLES]
    return _fewshot(system, examples, f"Диалог:\n{format_dialog(turns)}")


# =========================================================================== 2. extractor по разделу

EXTRACTOR_SYSTEM = """Ты — модуль извлечения фактов из диалога аптечного чата. Извлеки сведения ТОЛЬКО для раздела «{title}» (section="{section}"). Сведения других разделов игнорируй.

Общие правила:
- Извлекай только то, что сообщил покупатель. Не додумывай и не добавляй знания о препаратах и болезнях.
- Реплики ассистента — контекст: «нет» в ответ на вопрос ассистента — это ответ покупателя.
- Вопрос покупателя («можно ли мне X?») — не факт.
- subject — для кого сведения: "self", если о себе; иначе как назвал покупатель: "дочь", "сын", "мама", "ребёнок".
- evidence — дословная цитата из реплики покупателя, на которой основан факт.
- Поля, о которых покупатель не сказал, не включай. Других полей не добавляй.

Правила раздела:
{rules}

Ответ — только JSON без пояснений: {{"facts": [...], "section_statuses": [...]}}
Если сведений нет: {{"facts": [], "section_statuses": []}}"""

_NAME_RULE = ("- name — в начальной форме, но именно то, что назвал покупатель: торговое название не заменяй "
              "на действующее вещество, опечатки не исправляй.")
_POLARITY_RULE = ("- polarity: \"negative\", если покупатель говорит, что этого у него нет или он это не "
                  "принимает; иначе \"positive\".")

SECTION_RULES: dict[Section, str] = {
    Section.MEDICATIONS: "\n".join([
        "- Поля: section, subject, name, polarity, status, dosage, frequency, indication, evidence.",
        _NAME_RULE,
        "- status: \"active\" — принимает сейчас; \"prescribed\" — назначили, приём ещё не начат; "
        "\"paused\" — перерыв; \"stopped\" — бросил, перестал, отменили.",
        "- polarity \"negative\" — «X не принимаю», «X не пью».",
        "- dosage — дозировка, если названа отдельно от названия; frequency — как принимает; "
        "indication — от чего, если сказано.",
        "- «Никаких лекарств не принимаю» → section_statuses: status \"confirmed_absent\"; "
        "«не помню, что пью» → \"dont_know\".",
    ]),
    Section.ALLERGIES: "\n".join([
        "- Поля: section, subject, name, polarity, reaction_type, reaction, evidence.",
        "- name — аллерген в начальной форме, как назвал покупатель.",
        "- reaction_type: \"allergy\" — аллергия; \"intolerance\" — непереносимость, «плохо переношу».",
        "- reaction — как проявляется, если сказано.",
        "- «На X аллергии нет» → факт с polarity \"negative\".",
        "- «Аллергий нет» без уточнений → section_statuses: \"confirmed_absent\"; "
        "«не знаю, есть ли аллергия» → \"dont_know\".",
    ]),
    Section.CHRONIC_CONDITIONS: "\n".join([
        "- Поля: section, subject, name, polarity, status, evidence.",
        "- name — заболевание как назвал покупатель, в начальной форме.",
        "- status: \"active\" или \"remission\" (ремиссия).",
        _POLARITY_RULE,
        "- «Хронических болезней нет» → section_statuses: \"confirmed_absent\"; «не знаю» → \"dont_know\".",
    ]),
    Section.SPECIAL_STATES: "\n".join([
        "- Поля: section, subject, name, polarity, pregnancy_week, evidence.",
        "- name — строго одно из: \"беременность\", \"лактация\" (кормит грудью), \"планирование беременности\".",
        "- pregnancy_week — срок в неделях, только для беременности и только если назван.",
        "- «Не беременна», «не кормлю» → факт с polarity \"negative\".",
    ]),
    Section.DEMOGRAPHICS: "\n".join([
        "- Поля: section, subject, attribute, value, evidence. Поля name и polarity здесь нет.",
        "- attribute: \"age\" (лет), \"sex\" (\"м\" или \"ж\"), \"weight_kg\" (кг). Каждый атрибут — отдельный факт.",
        "- Пол указывай, только если он назван прямо («мужчина», «женщина»), не по окончаниям слов.",
    ]),
    Section.LIFESTYLE: "\n".join([
        "- Поля: section, subject, attribute, value, details, evidence. Поля name и polarity здесь нет.",
        "- attribute: \"smoking\" (курение), \"alcohol\" (алкоголь), \"driving\" (водит автомобиль).",
        "- value: \"да\", \"нет\" или \"иногда\". details — подробности, если сказаны.",
    ]),
    Section.SYMPTOMS: "\n".join([
        "- Поля: section, subject, name, polarity, status, severity, duration, evidence.",
        "- name — симптом своими словами покупателя, в начальной форме, без чисел («температура», «болит горло»).",
        "- status: \"active\" — беспокоит сейчас; \"resolved\" — прошло.",
        "- severity: \"лёгкая\", \"умеренная\" или \"сильная\" — только если покупатель так оценил.",
        "- duration — сколько длится, если сказано. «Кашля нет» → polarity \"negative\".",
    ]),
    Section.REPORTED_DIAGNOSES: "\n".join([
        "- Поля: section, subject, name, polarity, status, diagnosed_by, evidence.",
        "- name — диагноз как назвал покупатель. Хронические болезни сюда не относятся.",
        "- status: \"active\" или \"resolved\" (вылечили, прошло).",
        "- diagnosed_by: \"врач\" — если диагноз поставил врач; \"сам\" — если покупатель так считает сам.",
    ]),
}


def _fact(section: str, **kw: Any) -> dict:
    return {"section": section, **kw}


def _st(section: str, status: str, evidence: str, subject: str = "self") -> dict:
    return {"section": section, "subject": subject, "status": status, "evidence": evidence}


EXTRACTOR_EXAMPLES: dict[Section, list[tuple[list[dict[str, str]], dict]]] = {
    Section.MEDICATIONS: [
        (
            [{"role": "user", "content": "Пью Конкор 5 мг по утрам, а Омез бросил месяц назад. "
                                         "Врач ещё выписал Эутирокс, завтра начну."}],
            {"facts": [
                _fact("medications", subject="self", name="Конкор 5 мг", polarity="positive", status="active",
                      frequency="по утрам", evidence="Пью Конкор 5 мг по утрам"),
                _fact("medications", subject="self", name="Омез", polarity="positive", status="stopped",
                      evidence="а Омез бросил месяц назад"),
                _fact("medications", subject="self", name="Эутирокс", polarity="positive", status="prescribed",
                      evidence="Врач ещё выписал Эутирокс, завтра начну"),
            ], "section_statuses": []},
        ),
        (
            [{"role": "user", "content": "Аспирин я не принимаю. Можно ли мне Терафлю?"}],
            {"facts": [
                _fact("medications", subject="self", name="Аспирин", polarity="negative",
                      evidence="Аспирин я не принимаю"),
            ], "section_statuses": []},
        ),
    ],
    Section.ALLERGIES: [
        (
            [{"role": "user", "content": "На пенициллин у меня крапивница, лактозу плохо переношу. "
                                         "На аспирин аллергии нет."}],
            {"facts": [
                _fact("allergies", subject="self", name="пенициллин", polarity="positive", reaction_type="allergy",
                      reaction="крапивница", evidence="На пенициллин у меня крапивница"),
                _fact("allergies", subject="self", name="лактоза", polarity="positive", reaction_type="intolerance",
                      evidence="лактозу плохо переношу"),
                _fact("allergies", subject="self", name="аспирин", polarity="negative",
                      evidence="На аспирин аллергии нет"),
            ], "section_statuses": []},
        ),
        (
            [
                {"role": "assistant", "content": "Есть ли у дочки аллергия на лекарства?"},
                {"role": "user", "content": "Нет, никакой."},
            ],
            {"facts": [], "section_statuses": [_st("allergies", "confirmed_absent", "Нет, никакой.", "дочь")]},
        ),
    ],
    Section.CHRONIC_CONDITIONS: [
        (
            [{"role": "user", "content": "У меня диабет 2 типа. Язва была, но сейчас ремиссия."}],
            {"facts": [
                _fact("chronic_conditions", subject="self", name="диабет 2 типа", polarity="positive",
                      status="active", evidence="У меня диабет 2 типа"),
                _fact("chronic_conditions", subject="self", name="язва", polarity="positive", status="remission",
                      evidence="Язва была, но сейчас ремиссия"),
            ], "section_statuses": []},
        ),
        (
            [{"role": "user", "content": "Хронических болезней у мамы нет."}],
            {"facts": [], "section_statuses": [
                _st("chronic_conditions", "confirmed_absent", "Хронических болезней у мамы нет", "мама")]},
        ),
    ],
    Section.SPECIAL_STATES: [
        (
            [{"role": "user", "content": "Я на 12 неделе беременности."}],
            {"facts": [
                _fact("special_states", subject="self", name="беременность", polarity="positive", pregnancy_week=12,
                      evidence="Я на 12 неделе беременности"),
            ], "section_statuses": []},
        ),
        (
            [{"role": "user", "content": "Не беременна, но кормлю грудью."}],
            {"facts": [
                _fact("special_states", subject="self", name="беременность", polarity="negative",
                      evidence="Не беременна"),
                _fact("special_states", subject="self", name="лактация", polarity="positive",
                      evidence="кормлю грудью"),
            ], "section_statuses": []},
        ),
    ],
    Section.DEMOGRAPHICS: [
        (
            [{"role": "user", "content": "Дочке 6 лет, весит 20 кг. Мне самой 34."}],
            {"facts": [
                _fact("demographics", subject="дочь", attribute="age", value=6, evidence="Дочке 6 лет"),
                _fact("demographics", subject="дочь", attribute="weight_kg", value=20, evidence="весит 20 кг"),
                _fact("demographics", subject="self", attribute="age", value=34, evidence="Мне самой 34"),
            ], "section_statuses": []},
        ),
    ],
    Section.LIFESTYLE: [
        (
            [{"role": "user", "content": "Курю, пачку в день, за рулём каждый день. Алкоголь не пью."}],
            {"facts": [
                _fact("lifestyle", subject="self", attribute="smoking", value="да", details="пачку в день",
                      evidence="Курю, пачку в день"),
                _fact("lifestyle", subject="self", attribute="driving", value="да", details="каждый день",
                      evidence="за рулём каждый день"),
                _fact("lifestyle", subject="self", attribute="alcohol", value="нет", evidence="Алкоголь не пью"),
            ], "section_statuses": []},
        ),
    ],
    Section.SYMPTOMS: [
        (
            [{"role": "user", "content": "Третий день температура 38 и сильно болит горло. Кашля нет, "
                                         "насморк уже прошёл."}],
            {"facts": [
                _fact("symptoms", subject="self", name="температура", polarity="positive", status="active",
                      duration="третий день", evidence="Третий день температура 38"),
                _fact("symptoms", subject="self", name="болит горло", polarity="positive", status="active",
                      severity="сильная", evidence="сильно болит горло"),
                _fact("symptoms", subject="self", name="кашель", polarity="negative", evidence="Кашля нет"),
                _fact("symptoms", subject="self", name="насморк", polarity="positive", status="resolved",
                      evidence="насморк уже прошёл"),
            ], "section_statuses": []},
        ),
    ],
    Section.REPORTED_DIAGNOSES: [
        (
            [{"role": "user", "content": "Врач сказал, что у сына отит. А у меня, думаю, просто простуда."}],
            {"facts": [
                _fact("reported_diagnoses", subject="сын", name="отит", polarity="positive", status="active",
                      diagnosed_by="врач", evidence="Врач сказал, что у сына отит"),
                _fact("reported_diagnoses", subject="self", name="простуда", polarity="positive", status="active",
                      diagnosed_by="сам", evidence="у меня, думаю, просто простуда"),
            ], "section_statuses": []},
        ),
    ],
}


def extractor_messages(section: Section, turns: list[dict[str, str]]) -> Messages:
    policy = POLICIES[section]
    system = EXTRACTOR_SYSTEM.format(title=policy.title, section=section.value, rules=SECTION_RULES[section])
    examples = [(f"Диалог:\n{format_dialog(t)}", out) for t, out in EXTRACTOR_EXAMPLES[section]]
    return _fewshot(system, examples, f"Диалог:\n{format_dialog(turns)}")


# =========================================================================== 3. классификатор запроса

QUERY_DESCRIPTIONS: dict[QueryType, str] = {
    QueryType.SYMPTOM_RELIEF: "просит подобрать средство от симптома или болезни",
    QueryType.DRUG_COMPATIBILITY: "можно ли совмещать препараты между собой, с алкоголем или с заболеванием",
    QueryType.DOSING: "как принимать, сколько, как часто, дозировка для ребёнка",
    QueryType.ANALOG: "замена, аналог, подешевле, без побочного эффекта",
    QueryType.PREGNANCY_LACTATION: "можно ли при беременности или кормлении грудью",
    QueryType.THERAPY_HISTORY: "что покупатель принимал раньше",
    QueryType.PROFILE_REVIEW: "что ассистент о нём помнит",
    QueryType.GENERAL: "не связано со здоровьем покупателя: часы работы, наличие, цены, доставка",
}

QUERY_SYSTEM = """Ты — классификатор сообщений покупателя в аптечном чате. Определи тип последнего сообщения покупателя и о ком оно.

Типы (можно несколько):
{types}

subject — о ком вопрос: "self", если о себе или не указано; иначе как назвал покупатель: "дочь", "сын", "мама", "ребёнок".

Ответ — только JSON без пояснений: {{"query_types": [...], "subject": "..."}}"""

QUERY_EXAMPLES: list[tuple[str, dict]] = [
    ("Посоветуйте что-нибудь от изжоги, и можно ли это с Варфарином?",
     {"query_types": ["symptom_relief", "drug_compatibility"], "subject": "self"}),
    ("Сыну 10 лет, сколько Нурофена ему давать?", {"query_types": ["dosing"], "subject": "сын"}),
    ("Есть ли в наличии Смекта?", {"query_types": ["general"], "subject": "self"}),
]


def query_classifier_messages(message: str, history: list[dict[str, str]] | None = None) -> Messages:
    types = "\n".join(f"- {q.value} — {d}" for q, d in QUERY_DESCRIPTIONS.items())
    system = QUERY_SYSTEM.format(types=types)
    examples = [(f"Сообщение покупателя: {m}", out) for m, out in QUERY_EXAMPLES]
    ctx = ""
    if history:
        ctx = f"Предыдущие реплики:\n{format_dialog(history[-4:])}\n\n"
    return _fewshot(system, examples, f"{ctx}Сообщение покупателя: {message}")


# =========================================================================== 4. ответ ассистента

ANSWER_SYSTEM = """Ты — консультант аптеки в онлайн-чате. Отвечай по-русски, кратко и по делу.

Правила:
- Ты не врач и не ставишь диагнозы. Предлагай только безрецептурные средства и напоминай читать инструкцию.
- Учитывай блок ПАМЯТЬ: аллергии, принимаемые препараты, хронические заболевания, беременность и кормление, возраст.
- Если есть блок «НУЖНО УТОЧНИТЬ», сначала задай эти вопросы и не давай рекомендацию вслепую.
- Если сведения из памяти и из документированного источника расходятся, скажи об этом и уточни.
- Не придумывай сведения о покупателе, которых нет в памяти и в его сообщениях.
- При тревожных признаках (высокая температура дольше 3 дней, боль в груди, затруднённое дыхание, кровь, сильная боль, состояние ребёнка ухудшается) советуй обратиться к врачу или вызвать скорую.

{context}"""


def answer_messages(context: str, history: list[dict[str, str]], message: str) -> Messages:
    msgs: Messages = [{"role": "system", "content": ANSWER_SYSTEM.format(context=context)}]
    msgs += [{"role": t["role"], "content": t["content"]} for t in history]
    msgs.append({"role": "user", "content": message})
    return msgs
