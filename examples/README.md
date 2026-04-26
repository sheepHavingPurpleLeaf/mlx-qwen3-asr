# Examples

Practical workflows for running `mlx-qwen3-asr` on an Apple Silicon Mac.

## OpenAI-Compatible Server

Use your Mac as a local transcription API for tools that can target an
OpenAI-compatible base URL.

- [openai-server/README.md](openai-server/README.md)

## Real-World Workflows

Copy-paste commands for common jobs:

- subtitles from video/audio
- meeting transcription with speaker labels
- short/noisy scanner audio
- folder transcription

See [real-world/README.md](real-world/README.md).

## Batch Folder Script

Reuse one loaded model across many files:

```bash
python examples/batch-folder/batch_transcribe.py ./recordings --output-dir ./transcripts
```
