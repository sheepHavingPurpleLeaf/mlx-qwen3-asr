# Package Analytics

This directory documents lightweight visibility for `mlx-qwen3-asr` package
adoption.

## Quick Report

```bash
python scripts/pypi_download_stats.py
```

Write Markdown and JSON reports:

```bash
python scripts/pypi_download_stats.py \
  --markdown-output-file docs/analytics/pypi-downloads.md \
  --json-output-file docs/analytics/pypi-downloads.json
```

The script uses the public PyPIStats API, which is based on PyPI download logs.
It does not require secrets.

## Automated Reports

`.github/workflows/package-analytics.yml` runs weekly and can also be triggered
manually from the GitHub Actions UI. It uploads Markdown and JSON artifacts for
the latest download snapshot.

## What These Numbers Mean

PyPI download counts are download events, not unique humans or active
installations.

Useful signals:

- release spikes after publishing
- week-over-week trend
- Python version distribution
- operating system distribution

Known caveats:

- CI and automation can inflate downloads
- pip caches can reduce repeat downloads
- mirrors and private package caches can skew counts
- one real user can generate many download events

For deeper custom analysis, query the official public BigQuery table:
`bigquery-public-data.pypi.file_downloads`.
