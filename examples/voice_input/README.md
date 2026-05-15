# voice_input — macOS push-to-talk transcription

Hold **Fn** anywhere on macOS → speak → release → transcript is auto-pasted at the cursor.

## Install

From the repo root, with the project venv active:

```bash
uv pip install sounddevice pyobjc-framework-Quartz pyobjc-framework-Cocoa rumps
```

The model must be at `models/Qwen3-ASR-0.6B/` (download Qwen/Qwen3-ASR-0.6B from HuggingFace).

### Optional: inverse text normalization (ITN)

Turns spoken-form numbers/dates into digits — `"一点三四"` → `"1.34"`,
`"二零二六年五月五日"` → `"2026/05/05"`, `"百分之二十"` → `"20%"`.
Enabled by default for Chinese; falls back silently if not installed.

```bash
brew install openfst
CPPFLAGS="-I$(brew --prefix openfst)/include" \
LDFLAGS="-L$(brew --prefix openfst)/lib" \
CXXFLAGS="-std=c++17 -I$(brew --prefix openfst)/include" \
uv pip install pynini
uv pip install --no-deps WeTextProcessing
uv pip install importlib_resources
```

`--no-deps` is required: WeTextProcessing pins `pynini==2.1.6`, but Homebrew's
OpenFst 1.8.4 renamed `StringJoin → StrJoin` and only pynini ≥2.1.7 builds
against it. The two are ABI-compatible at runtime.

To disable ITN, construct the engine with `Engine(itn=False)` in `app.py`.

## Run

```bash
python -m examples.voice_input.app
```

A 🎤 will appear in the menu bar. Wait for the status to read "idle" (model load + warmup; ~6 s the first time, ~1.5 s on subsequent runs).

## Required permissions (macOS)

The app will fail silently or crash on first launch unless these are granted in **System Settings → Privacy & Security**:

| Permission | Why |
|---|---|
| **Microphone** | Audio capture (sounddevice). System auto-prompts on first capture. |
| **Input Monitoring** | Detect Fn key globally (Quartz CGEventTap). Add the Python interpreter (or your terminal) here manually. |
| **Accessibility** | Synthesize Cmd+V to paste. Add the same process here. |

Tip: the binary that needs the permissions is the *Python interpreter* you launch with — usually `.venv/bin/python` — not "Python" generically. After the first crash/permission denial, drag the exact binary into the privacy lists.

## Behaviour

- Default language: **Chinese**. Edit `app.py` or `engine.py` to change.
- Holds shorter than 300 ms are discarded (avoid stray Fn taps).
- Output is set on the pasteboard, then a Cmd+V is synthesized — your previous clipboard is overwritten.

## Limitations / known issues

- **Streaming is not used**: each press records the full utterance, then transcribes once. Latency is ~1–2 s after release for typical 5 s clips.
- Some external keyboards do not emit `NSEventModifierFlagFunction` reliably. Internal MacBook keyboards work.
- Cmd+V can be swallowed by apps that don't accept synthetic events (rare).
