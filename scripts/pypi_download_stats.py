#!/usr/bin/env python3
"""Fetch PyPI download visibility for a package via the PyPIStats API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_PACKAGE = "mlx-qwen3-asr"
DEFAULT_API_BASE = "https://pypistats.org/api/packages"
USER_AGENT = "mlx-qwen3-asr-download-stats/1.0"


@dataclass(frozen=True)
class Report:
    package: str
    generated_at: str
    mirrors: bool
    days: int
    recent: dict[str, int]
    daily: list[dict[str, Any]]
    totals: dict[str, int]
    breakdowns: dict[str, dict[str, int]]


def _fetch_json(url: str, retries: int = 4) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                retry_after = exc.headers.get("Retry-After")
                delay = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                time.sleep(delay)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError(f"Failed to fetch {url}")


def _endpoint(api_base: str, package: str, name: str, params: dict[str, str] | None = None) -> str:
    quoted = urllib.parse.quote(package)
    url = f"{api_base.rstrip('/')}/{quoted}/{name}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return url


def _parse_day(row: dict[str, Any]) -> date:
    value = row.get("date")
    if not isinstance(value, str):
        raise ValueError(f"Missing date in row: {row!r}")
    return date.fromisoformat(value)


def _filter_days(rows: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    last_day = max(_parse_day(row) for row in rows)
    first_day = last_day - timedelta(days=days - 1)
    return [row for row in rows if _parse_day(row) >= first_day]


def _sum_rows(rows: list[dict[str, Any]]) -> int:
    return sum(int(row.get("downloads", 0)) for row in rows)


def _sum_last_days(rows: list[dict[str, Any]], days: int) -> int:
    return _sum_rows(_filter_days(rows, days))


def _category_totals(rows: list[dict[str, Any]], days: int) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for row in _filter_days(rows, days):
        category = str(row.get("category", "unknown"))
        totals[category] += int(row.get("downloads", 0))
    return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))


def collect_report(
    package: str,
    *,
    days: int,
    mirrors: bool,
    api_base: str = DEFAULT_API_BASE,
    include_breakdowns: bool = True,
) -> Report:
    mirror_param = {"mirrors": "true" if mirrors else "false"}

    recent_raw = _fetch_json(_endpoint(api_base, package, "recent"))
    daily_raw = _fetch_json(_endpoint(api_base, package, "overall", mirror_param))

    daily = list(daily_raw.get("data", []))
    recent = {key: int(value) for key, value in recent_raw.get("data", {}).items()}
    selected_daily = _filter_days(daily, days)

    totals = {
        "all_time": _sum_rows(daily),
        "selected_days": _sum_rows(selected_daily),
        "last_7_days": _sum_last_days(daily, 7),
        "last_30_days": _sum_last_days(daily, 30),
    }

    breakdowns: dict[str, dict[str, int]] = {}
    if include_breakdowns:
        for name in ("python_major", "python_minor", "system"):
            raw = _fetch_json(_endpoint(api_base, package, name, mirror_param))
            breakdowns[name] = _category_totals(list(raw.get("data", [])), days)

    return Report(
        package=package,
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        mirrors=mirrors,
        days=days,
        recent=recent,
        daily=selected_daily,
        totals=totals,
        breakdowns=breakdowns,
    )


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def render_markdown(report: Report) -> str:
    mirror_label = "including mirrors" if report.mirrors else "excluding mirrors"
    lines = [
        f"# PyPI Download Stats: {report.package}",
        "",
        f"- Generated: `{report.generated_at}`",
        "- Source: PyPIStats API, backed by PyPI download logs",
        f"- Mirror mode: `{mirror_label}`",
        f"- Selected window: last `{report.days}` days available in the daily series",
        "",
        "Download counts are events, not unique users. CI, repeated installs, caches,",
        "mirrors, and automated systems can skew the numbers.",
        "",
        "## Recent Snapshot",
        "",
        _markdown_table(
            ["Window", "Downloads"],
            [
                ["Last day", _fmt_int(report.recent.get("last_day", 0))],
                ["Last week", _fmt_int(report.recent.get("last_week", 0))],
                ["Last month", _fmt_int(report.recent.get("last_month", 0))],
            ],
        ),
        "",
        "## Daily Series Summary",
        "",
        _markdown_table(
            ["Window", "Downloads"],
            [
                ["All available daily rows", _fmt_int(report.totals["all_time"])],
                [f"Selected last {report.days} days", _fmt_int(report.totals["selected_days"])],
                ["Last 30 days", _fmt_int(report.totals["last_30_days"])],
                ["Last 7 days", _fmt_int(report.totals["last_7_days"])],
            ],
        ),
        "",
    ]

    if report.daily:
        lines.extend(
            [
                "## Recent Daily Downloads",
                "",
                _markdown_table(
                    ["Date", "Downloads"],
                    [
                        [str(row["date"]), _fmt_int(int(row.get("downloads", 0)))]
                        for row in report.daily[-14:]
                    ],
                ),
                "",
            ]
        )

    for name, totals in report.breakdowns.items():
        title = name.replace("_", " ").title()
        rows = [[category, _fmt_int(count)] for category, count in list(totals.items())[:12]]
        if not rows:
            continue
        lines.extend(
            [
                f"## {title} Breakdown",
                "",
                _markdown_table(["Category", f"Downloads in last {report.days} days"], rows),
                "",
            ]
        )

    lines.extend(
        [
            "## Deeper Analysis",
            "",
            "For exact custom cuts, query the public BigQuery table:",
            "`bigquery-public-data.pypi.file_downloads`.",
            "",
        ]
    )
    return "\n".join(lines)


def report_to_json(report: Report) -> dict[str, Any]:
    return {
        "package": report.package,
        "generated_at": report.generated_at,
        "mirrors": report.mirrors,
        "days": report.days,
        "recent": report.recent,
        "daily": report.daily,
        "totals": report.totals,
        "breakdowns": report.breakdowns,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=DEFAULT_PACKAGE, help="PyPI package name")
    parser.add_argument("--days", type=int, default=90, help="Daily/breakdown lookback window")
    parser.add_argument("--mirrors", action="store_true", help="Include mirror downloads")
    parser.add_argument(
        "--no-breakdowns",
        action="store_true",
        help="Skip Python and system breakdown endpoints",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    parser.add_argument("--output-file", type=Path, help="Write report to this file")
    parser.add_argument(
        "--markdown-output-file",
        type=Path,
        help="Also write a Markdown report to this file from the same API snapshot",
    )
    parser.add_argument(
        "--json-output-file",
        type=Path,
        help="Also write a JSON report to this file from the same API snapshot",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.days < 1:
        raise SystemExit("--days must be >= 1")

    report = collect_report(
        args.package,
        days=args.days,
        mirrors=args.mirrors,
        api_base=args.api_base,
        include_breakdowns=not args.no_breakdowns,
    )

    if args.format == "json":
        output = json.dumps(report_to_json(report), indent=2, sort_keys=True) + "\n"
    else:
        output = render_markdown(report)

    if args.markdown_output_file:
        args.markdown_output_file.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output_file.write_text(render_markdown(report), encoding="utf-8")
    if args.json_output_file:
        args.json_output_file.parent.mkdir(parents=True, exist_ok=True)
        args.json_output_file.write_text(
            json.dumps(report_to_json(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(output, encoding="utf-8")
    else:
        print(output, end="" if output.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
