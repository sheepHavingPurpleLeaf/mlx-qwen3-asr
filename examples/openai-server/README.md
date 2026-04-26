# OpenAI-Compatible Local Server

Run Qwen3-ASR on your Mac and point OpenAI-compatible clients at it.

## Start The Server

```bash
pip install "mlx-qwen3-asr[serve]"
export MLX_ASR_API_KEY="$(openssl rand -hex 16)"
mlx-qwen3-asr serve --api-key "$MLX_ASR_API_KEY"
```

The default base URL is:

```text
http://localhost:8765/v1
```

## Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_KEY",
    base_url="http://localhost:8765/v1",
)

result = client.audio.transcriptions.create(
    model="Qwen/Qwen3-ASR-0.6B",
    file=open("recording.wav", "rb"),
)

print(result.text)
```

Model discovery also works:

```python
for model in client.models.list():
    print(model.id)
```

## cURL

```bash
curl http://localhost:8765/v1/audio/transcriptions \
  -H "Authorization: Bearer $MLX_ASR_API_KEY" \
  -F "file=@recording.wav" \
  -F "model=Qwen/Qwen3-ASR-0.6B"
```

Verbose JSON with word timestamps:

```bash
curl http://localhost:8765/v1/audio/transcriptions \
  -H "Authorization: Bearer $MLX_ASR_API_KEY" \
  -F "file=@recording.wav" \
  -F "model=Qwen/Qwen3-ASR-0.6B" \
  -F "response_format=verbose_json"
```

Subtitle output:

```bash
curl http://localhost:8765/v1/audio/transcriptions \
  -H "Authorization: Bearer $MLX_ASR_API_KEY" \
  -F "file=@video.mp4" \
  -F "model=Qwen/Qwen3-ASR-0.6B" \
  -F "response_format=srt" \
  > subtitles.srt
```

## JavaScript Fetch

```javascript
const form = new FormData();
form.append("model", "Qwen/Qwen3-ASR-0.6B");
form.append("file", fileInput.files[0]);

const response = await fetch("http://localhost:8765/v1/audio/transcriptions", {
  method: "POST",
  headers: { Authorization: `Bearer ${apiKey}` },
  body: form,
});

const data = await response.json();
console.log(data.text);
```

## When To Use The Async API

`/v1/audio/transcriptions` is synchronous and best for short clips or SDK
compatibility. For long recordings, use `/transcribe` plus `/jobs/{job_id}` to
avoid client HTTP timeouts.
