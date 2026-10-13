"""Промпты: примеры проходят ту же валидацию, что и ответы модели; extractor видит только свой раздел."""
import pytest

from pharmaltm import QueryClassification, Section, parse_extraction
from pharmaltm.prompts import (
    CLASSIFIER_EXAMPLES,
    EXTRACTOR_EXAMPLES,
    QUERY_EXAMPLES,
    SECTION_RULES,
    extractor_messages,
    format_dialog,
    query_classifier_messages,
    section_classifier_messages,
)

TURNS = [{"role": "user", "content": "Пью Энап"}, {"role": "assistant", "content": "Понял"}]


@pytest.mark.parametrize("section", list(Section))
def test_every_section_has_rules_and_examples(section):
    assert SECTION_RULES[section] and EXTRACTOR_EXAMPLES[section]


@pytest.mark.parametrize("section", list(Section))
def test_extractor_examples_are_valid_and_on_section(section):
    for _, out in EXTRACTOR_EXAMPLES[section]:
        facts, statuses, rejected = parse_extraction(out)
        assert not rejected, rejected
        assert all(x.section == section.value for x in [*facts, *statuses])


def test_classifier_examples_use_known_sections():
    for _, sections in CLASSIFIER_EXAMPLES:
        assert all(Section(s) for s in sections)


def test_query_examples_are_valid():
    for _, out in QUERY_EXAMPLES:
        QueryClassification.model_validate(out)


@pytest.mark.parametrize("section", list(Section))
def test_extractor_prompt_contains_only_its_section_rules(section):
    system = extractor_messages(section, TURNS)[0]["content"]
    assert f'section="{section.value}"' in system
    for other, rules in SECTION_RULES.items():
        if other != section:
            assert rules not in system


def test_fewshot_structure():
    for msgs in (section_classifier_messages(TURNS), extractor_messages(Section.ALLERGIES, TURNS),
                 query_classifier_messages("Что выпить от головы?")):
        assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"
        roles = [m["role"] for m in msgs[1:-1]]
        assert roles == ["user", "assistant"] * (len(roles) // 2)


def test_dialog_formatting():
    assert format_dialog(TURNS) == "Покупатель: Пью Энап\nАссистент: Понял"
    assert section_classifier_messages(TURNS)[-1]["content"].endswith("Ассистент: Понял")


def test_query_prompt_includes_recent_history():
    msgs = query_classifier_messages("А ребёнку можно?", history=TURNS)
    assert "Предыдущие реплики" in msgs[-1]["content"] and "Пью Энап" in msgs[-1]["content"]
