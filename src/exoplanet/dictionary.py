"""Parse the official NASA Exoplanet Archive KOI and TCE column pages."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path

COLUMN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _ColumnTableParser(HTMLParser):
    """Collect text from h2 headings and from tr.column rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.section = ""
        self._in_h2 = False
        self._in_row = False
        self._in_cell = False
        self._buf: list[str] = []
        self._cells: list[str] = []
        self.rows: list[tuple[str, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        if tag == "h2":
            self._in_h2 = True
            self._buf = []
        if tag == "tr" and "column" in attr.get("class", "").split():
            self._in_row = True
            self._cells = []
        if self._in_row and tag == "td":
            self._in_cell = True
            self._buf = []
        if self._in_cell and tag == "br":
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self._in_h2:
            self.section = _collapse(self._buf)
            self._in_h2 = False
        if tag == "td" and self._in_cell:
            self._cells.append(_collapse(self._buf))
            self._in_cell = False
        if tag == "tr" and self._in_row:
            if self._cells:
                self.rows.append((self.section, self._cells))
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_h2 or self._in_cell:
            self._buf.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._in_h2 or self._in_cell:
            self._buf.append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if self._in_h2 or self._in_cell:
            self._buf.append(html.unescape(f"&#{name};"))


def _collapse(parts: list[str]) -> str:
    text = html.unescape("".join(parts)).replace("\xa0", " ")
    return " ".join(text.split())


def _read_html(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "iso-8859-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("iso-8859-1", errors="replace")


def _names(cell: str) -> list[str]:
    cleaned = cell.replace("†", " ").replace("*", " ")
    names = []
    for part in cleaned.split(","):
        name = part.strip()
        if COLUMN_NAME.fullmatch(name):
            names.append(name)
    return names


def parse_column_page(path: Path) -> dict[str, dict]:
    """Return one record per database column named on an archive column page."""
    parser = _ColumnTableParser()
    parser.feed(_read_html(path))
    columns: dict[str, dict] = {}
    for section, cells in parser.rows:
        names = _names(cells[0]) if cells else []
        label = cells[1] if len(cells) > 1 else ""
        description = cells[2] if len(cells) > 2 else ""
        uncertainty_names = _names(cells[3]) if len(cells) > 3 else []
        for name in names:
            columns[name] = {
                "column": name,
                "section": section,
                "label": label,
                "description": description,
                "kind": "measurement",
                "parent": "",
                "uncertainty_columns": uncertainty_names,
                "source_file": path.name,
            }
        for err_name in uncertainty_names:
            columns[err_name] = {
                "column": err_name,
                "section": section,
                "label": f"Uncertainty on {label}",
                "description": (
                    f"Named on the archive column page as an uncertainty column "
                    f"for {', '.join(names)}. {description}"
                ).strip(),
                "kind": "uncertainty",
                "parent": names[0] if names else "",
                "uncertainty_columns": [],
                "source_file": path.name,
            }
    return columns


def odd_even_documentation(tce_columns: dict[str, dict]) -> dict:
    """Documentation for koi_bin_oedp_sig, which the KOI page does not list."""
    record = tce_columns.get("tce_bin_oedp_sig", {})
    return {
        "column": "koi_bin_oedp_sig",
        "section": "TCE / light-curve vetting statistic",
        "label": record.get("label", ""),
        "description": record.get("description", ""),
        "kind": "measurement",
        "parent": "",
        "uncertainty_columns": [],
        "source_file": record.get("source_file", ""),
        "documentation_note": (
            "koi_bin_oedp_sig is not listed on the KOI cumulative column page "
            "retrieved 2026-10-31. The TCE column page defines tce_bin_oedp_sig "
            "with this description and states that the KOI table reports the "
            "same odd-even comparison as a sigma deviation. The 2012 Kepler "
            "table-definitions document describes that KOI statistic without "
            "using the koi_bin_oedp_sig name."
        ),
    }
