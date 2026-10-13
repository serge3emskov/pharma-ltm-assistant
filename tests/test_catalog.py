"""Справочник препаратов: разбор выгрузки ГРЛС и слияние со стартовым набором."""
from __future__ import annotations

import csv
import io
import zipfile

import pytest

from pharmaltm import Normalizer, Section
from pharmaltm.catalog import (
    DrugCatalog,
    Product,
    clean_trade_name,
    detect_columns,
    find_header,
    load_grls,
    parse_atc,
    parse_inn,
)

openpyxl = pytest.importorskip("openpyxl")

HEADER = ["Регистрационный номер", "Торговое наименование лекарственного препарата",
          "Международное непатентованное или группировочное или химическое наименование",
          "Лекарственная форма", "Анатомо-терапевтическая химическая классификация (АТХ)", "Дата аннулирования"]

ROWS = [
    ["ЛП-000001", "Нурофен®", "Ибупрофен", "таблетки", "M01AE01", None],
    ["ЛП-000002", "ТераФлю от гриппа и простуды", "Парацетамол+Фенилэфрин+Фенирамин", "порошок", "N02BE51", None],
    ["ЛП-000003", "Ингавирин", "Имидазолилэтанамид пентандиовой кислоты", "капсулы", "J05AX", None],
    ["ЛП-000004", "Кагоцел", "~", "таблетки", "J05AX", None],
    ["ЛП-000005", "Вольтарен Эмульгель", "Диклофенак", "гель", "M02AA15", None],
    ["ЛП-000006", "Старый препарат", "Ибупрофен", "таблетки", "M01AE01", "01.02.2020"],
    ["ЛП-000007", None, "Ибупрофен", "таблетки", "M01AE01", None],
]


def _xlsx(rows, title_rows=2, sheets=None) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Действующий"
    for _ in range(title_rows):
        ws.append(["Государственный реестр лекарственных средств на 01.10.2026"])
    ws.append(HEADER)
    for r in rows:
        ws.append(r)
    for name, extra in (sheets or {}).items():
        sh = wb.create_sheet(name)
        sh.append(HEADER)
        for r in extra:
            sh.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- разбор ячеек


@pytest.mark.parametrize("cell, expected", [
    ("N02BE01", ("N02BE01",)),
    ("N02BE01 Парацетамол", ("N02BE01",)),
    ("М01АЕ01", ("M01AE01",)),                 # кириллица вместо латиницы
    ("N02BE51; R05X", ("N02BE51", "R05X")),
    ("Витамин B12", ()),                       # код в названии витамина — не АТХ
    ("B03", ("B03",)),
    (None, ()),
    ("", ()),
])
def test_parse_atc(cell, expected):
    assert parse_atc(cell) == expected


@pytest.mark.parametrize("cell, expected", [
    ("Ибупрофен", ("ибупрофен",)),
    ("Амоксициллин+[Клавулановая кислота]", ("амоксициллин", "клавулановая кислота")),
    ("Парацетамол + Фенилэфрин + Фенирамин", ("парацетамол", "фенилэфрин", "фенирамин")),
    ("Смектит диоктаэдрический", ("диосмектит",)),  # синоним МНН из стартового набора
    ("Гепарин натрия (в виде соли)", ("гепарин натрия",)),
    ("~", ()),
    ("-", ()),
    (None, ()),
])
def test_parse_inn(cell, expected):
    assert parse_inn(cell) == expected


def test_clean_trade_name():
    assert clean_trade_name("«Нурофен®  Экспресс»") == "Нурофен Экспресс"


# ---------------------------------------------------------------- поиск колонок


def test_detect_columns_by_keywords():
    cols = detect_columns(HEADER)
    assert cols == {"reg_number": 0, "trade_name": 1, "inn": 2, "form": 3, "atc": 4, "annulled": 5}


def test_find_header_below_title_rows_and_split_header():
    rows = [["Реестр на 01.10.2026"], [], ["Торговое", "Международное непатентованное", "Код"],
            ["наименование", "наименование", "АТХ"], ["Нурофен", "Ибупрофен", "M01AE01"]]
    start, n, cols = find_header(rows)
    assert (start, n) == (2, 2) and cols["trade_name"] == 0 and cols["inn"] == 1 and cols["atc"] == 2


def test_manual_columns_when_headers_unknown(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text("Name;Substance;Code\nНурофен;Ибупрофен;M01AE01\n", encoding="utf-8")
    products, report = load_grls(path)
    assert not products and "не найдены колонки" in report.tables[0].reason
    products, report = load_grls(path, columns={"trade_name": "Name", "inn": "Substance", "atc": "Code"})
    assert products == [Product("Нурофен", ("ибупрофен",), ("M01AE01",))]


# ---------------------------------------------------------------- форматы файлов


def test_load_xlsx_skips_annulled_rows_and_excluded_sheet(tmp_path):
    path = tmp_path / "grls.xlsx"
    path.write_bytes(_xlsx(ROWS, sheets={"Исключенные": [["ЛП-9", "Удалённый", "Ибупрофен", "", "M01AE01", None]]}))
    products, report = load_grls(path)
    names = [p.trade_name for p in products]
    assert names == ["Нурофен", "ТераФлю от гриппа и простуды", "Ингавирин", "Кагоцел", "Вольтарен Эмульгель"]
    main, excluded = report.tables
    assert main.used and main.header_row == 2 and main.skipped == {"аннулировано": 1, "нет торгового наименования": 1}
    assert not excluded.used and excluded.reason == "таблица исключённых записей"
    assert products[3].components == () and products[1].atc == ("N02BE51",)
    assert report.as_dict()["products"] == 5


def test_load_zip_with_xlsx_inside(tmp_path):
    path = tmp_path / "grls.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("реестр.xlsx", _xlsx(ROWS[:2], title_rows=0))
        zf.writestr("readme.txt.bak", "не таблица")
    products, report = load_grls(path)
    assert [p.trade_name for p in products] == ["Нурофен", "ТераФлю от гриппа и простуды"]
    assert report.tables[0].name.startswith("реестр.xlsx")


def test_load_csv_cp1251_semicolon(tmp_path):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(HEADER)
    w.writerow(ROWS[0])
    path = tmp_path / "grls.csv"
    path.write_bytes(buf.getvalue().encode("cp1251"))
    products, _ = load_grls(path)
    assert products[0] == Product("Нурофен", ("ибупрофен",), ("M01AE01",), "ЛП-000001", "таблетки")


def test_load_xls(tmp_path):
    xlwt = pytest.importorskip("xlwt")
    pytest.importorskip("xlrd")
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Реестр")
    for c, v in enumerate(HEADER):
        ws.write(0, c, v)
    for r, row in enumerate(ROWS[:3], start=1):
        for c, v in enumerate(row):
            if v is not None:
                ws.write(r, c, v)
    path = tmp_path / "grls.xls"
    wb.save(str(path))
    products, _ = load_grls(path)
    assert [p.trade_name for p in products] == ["Нурофен", "ТераФлю от гриппа и простуды", "Ингавирин"]


# ---------------------------------------------------------------- справочник


@pytest.fixture(scope="module")
def catalog(tmp_path_factory) -> DrugCatalog:
    path = tmp_path_factory.mktemp("grls") / "grls.xlsx"
    path.write_bytes(_xlsx(ROWS))
    cat, _ = DrugCatalog.from_grls(path)
    cat.add_products([
        Product("Колдрекс", ("парацетамол", "фенилэфрин"), ("N02BE51",)),
        Product("Колдрекс", ("парацетамол", "кофеин"), ("N02BE51",)),     # одно наименование — разный состав
        Product("Парацетамол", ("парацетамол", "кофеин"), ("N02BE51",)),  # МНН важнее чужого торгового
        Product("Энап", ("эналаприл",), ("C09AA02",)),
        Product("Норма", ("ибупрофен",), ("M01AE01",)),                   # обычное слово
    ])
    return cat


def test_grls_merges_into_seed_entries(catalog):
    assert catalog.resolve("Энап 5 мг").canonical == "эналаприл"  # ключи памяти стартового набора не меняются
    teraflu = catalog.resolve("ТераФлю от гриппа и простуды")
    assert teraflu.canonical == "парацетамол + фенилэфрин + фенирамин"  # тот же состав в другом порядке
    assert catalog.resolve("Терафлю").canonical == teraflu.canonical
    summary = catalog.summary()
    assert summary["entries_by_source"]["grls+seed"] >= 3 and summary["products"] == 10


def test_new_drugs_from_grls(catalog):
    ing = catalog.resolve("Ингавирин 90 мг")
    assert ing.canonical == "имидазолилэтанамид пентандиовой кислоты" and ing.atc == ("J05AX",)
    kag = catalog.resolve("Кагоцел")
    assert kag.matched and kag.canonical == "кагоцел"  # без МНН ключ — торговое наименование
    assert [m.canonical for m in catalog.mentions("Пью Ингавирин и Кагоцел третий день")] == [
        "имидазолилэтанамид пентандиовой кислоты", "кагоцел"]


def test_ambiguous_trade_name_is_not_guessed(catalog):
    assert not catalog.resolve("Колдрекс").matched
    assert ("колдрекс", ["парацетамол + кофеин", "парацетамол + фенилэфрин"]) in catalog.ambiguous()
    assert catalog.resolve("Парацетамол").canonical == "парацетамол"


def test_common_words_are_not_mentions(catalog):
    assert catalog.resolve("Норма").canonical == "ибупрофен"  # как название — находится
    assert catalog.mentions("Давление в норме, всё нормально") == []  # в речи — нет


def test_trade_name_atc_for_topical_forms(catalog):
    assert catalog.resolve("Вольтарен Эмульгель").atc == ("M02AA15",)
    assert "M01AB05" in catalog.resolve("Диклофенак").atc


def test_normalizer_from_catalog(catalog):
    n = Normalizer.from_catalog(catalog)
    assert n.entity(Section.MEDICATIONS, "Ингавирин").normalized == "имидазолилэтанамид пентандиовой кислоты"
    assert n.entity(Section.MEDICATIONS, "Лозап 50").normalized == "лозартан"
    assert n.entity(Section.ALLERGIES, "Ингавирин").normalized == "имидазолилэтанамид пентандиовой кислоты"
    assert n.entity(Section.CHRONIC_CONDITIONS, "диабет").normalized == "сахарный диабет"


def test_save_and_load(catalog, tmp_path):
    path = tmp_path / "catalog.json.gz"
    catalog.save(path)
    loaded = DrugCatalog.load(path)
    assert loaded.summary() == catalog.summary()
    for name in ("Энап", "Ингавирин", "Вольтарен Эмульгель", "Колдрекс", "Терафлю"):
        assert loaded.resolve(name) == catalog.resolve(name)
    text = "Пью Ингавирин, давление в норме"
    assert loaded.mentions(text) == catalog.mentions(text)


def test_seed_catalog_is_default():
    n = Normalizer()
    assert len(n.catalog) == len(DrugCatalog()) and n.catalog.summary()["products"] == 0
    assert n.entity(Section.MEDICATIONS, "Вольтарен Эмульгель").normalized == "диклофенак"
