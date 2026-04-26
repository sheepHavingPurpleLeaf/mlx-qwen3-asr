# Real-World Workflows

These examples are intentionally plain commands. They are meant to be copied,
changed, and run from the shell.

## Subtitles

Create SRT subtitles from a video or audio file:

```bash
mlx-qwen3-asr video.mp4 -f srt -o subtitles/
```

Create WebVTT:

```bash
mlx-qwen3-asr video.mp4 -f vtt -o subtitles/
```

Subtitle formats automatically enable word timestamps in offline mode.

## Meeting Transcription

Plain meeting transcript:

```bash
mlx-qwen3-asr meeting.m4a -f json -o transcripts/
```

Speaker-labeled transcript:

```bash
pip install "mlx-qwen3-asr[diarize]"
export PYANNOTE_AUTH_TOKEN=hf_...
mlx-qwen3-asr meeting.m4a --diarize --num-speakers 2 -f json -o transcripts/
```

Use `--min-speakers` and `--max-speakers` when the speaker count is unknown.

## Short Or Noisy Scanner Audio

For short noisy clips, keep JSON output so callers can inspect decode stop
metadata:

```bash
mlx-qwen3-asr scanner_clip.wav -f json -o scanner-out/
```

The JSON includes fields such as `finish_reason`, `truncated`,
`generated_tokens`, and `max_new_tokens`. If `truncated` is true for useful
speech, retry with an explicit larger cap:

```bash
mlx-qwen3-asr scanner_clip.wav -f json --max-new-tokens 256 -o scanner-out/
```

## Batch Folder

For a quick shell-only batch:

```bash
mkdir -p transcripts
find recordings -type f \( -name '*.wav' -o -name '*.mp3' -o -name '*.m4a' \) \
  -print0 |
while IFS= read -r -d '' file; do
  mlx-qwen3-asr "$file" -f json -o transcripts/
done
```

For larger folders, prefer the Python helper so the model loads once:

```bash
python examples/batch-folder/batch_transcribe.py recordings \
  --output-dir transcripts \
  --output-format json \
  --skip-existing
```

## Local API

When another tool supports an OpenAI-compatible transcription endpoint, run:

```bash
export MLX_ASR_API_KEY="$(openssl rand -hex 16)"
mlx-qwen3-asr serve --api-key "$MLX_ASR_API_KEY"
```

Then configure the tool with:

```text
Base URL: http://localhost:8765/v1
API key:  the value of MLX_ASR_API_KEY
Model:    Qwen/Qwen3-ASR-0.6B
```
