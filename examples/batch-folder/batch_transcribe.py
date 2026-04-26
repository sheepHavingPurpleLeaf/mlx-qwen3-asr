#!/usr/bin/env python3
"""Batch-transcribe a folder while reusing one loaded model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlx_qwen3_asr import Session
from mlx_qwen3_asr.writers import get_writer

DEFAULT_PATTERNS = ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.opus", "*.mp4", "*.mov")


def _discover_inputs(root: Path, patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    if root.is_file():
        return [root]
    for pattern in patterns:
        files.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(files)


def _output_path(audio_path: Path, *, input_root: Path, output_dir: Path, suffix: str) -> Path:
    if input_root.is_file():
        rel = audio_path.name
    else:
        rel = audio_path.relative_to(input_root)
    return (output_dir / rel).with_suffix(f".{suffix}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Audio/video file or folder")
    parser.add_argument("--output-dir", type=Path, default=Path("transcripts"))
    parser.add_argument(
        "--output-format",
        choices=("txt", "json", "srt", "vtt", "tsv"),
        default="json",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--language", help="Optional language hint, e.g. English")
    parser.add_argument("--context", default="", help="Optional vocabulary/domain context")
    parser.add_argument("--timestamps", action="store_true", help="Force word timestamps")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--glob",
        action="append",
        dest="patterns",
        help="Input glob pattern. Can be repeated. Defaults cover common media files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    input_root = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    patterns = args.patterns or list(DEFAULT_PATTERNS)

    if not input_root.exists():
        print(f"Input not found: {input_root}", file=sys.stderr)
        return 1

    files = _discover_inputs(input_root, patterns)
    if not files:
        print(f"No matching media files found under: {input_root}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = get_writer(args.output_format)
    need_timestamps = args.timestamps or args.output_format in {"srt", "vtt"}

    print(f"Loading {args.model} once for {len(files)} file(s)...", file=sys.stderr)
    session = Session(model=args.model)

    completed = 0
    skipped = 0
    for index, audio_path in enumerate(files, start=1):
        out_path = _output_path(
            audio_path,
            input_root=input_root,
            output_dir=output_dir,
            suffix=args.output_format,
        )
        if args.skip_existing and out_path.exists():
            skipped += 1
            print(f"[{index}/{len(files)}] skip {audio_path}", file=sys.stderr)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(files)}] transcribe {audio_path}", file=sys.stderr)
        result = session.transcribe(
            str(audio_path),
            language=args.language,
            context=args.context,
            return_timestamps=need_timestamps,
            return_chunks=args.output_format == "json",
        )
        writer(result, str(out_path))
        completed += 1
        if result.truncated:
            print(
                f"  warning: decode truncated; inspect {out_path}",
                file=sys.stderr,
            )

    print(
        f"Done: {completed} written, {skipped} skipped, output={output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
