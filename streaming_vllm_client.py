# coding=utf-8
import io
import json
import time
import asyncio
import argparse
import websockets
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, List

SERVER_URL = "ws://118.89.233.123:18001"

def _read_audio_bytes(wav_path: str) -> bytes:
    with open(wav_path, 'rb') as f:
        return f.read()

def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


async def test_streaming_vllm_ws(
    wav16k: np.ndarray,
    step_ms: int = 500,
    hotwords: List[str] = None,
    city: str = None
) -> dict:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))
    hotwords = hotwords or []

    uri = f"{SERVER_URL}/ws/asr"
    print(f"连接服务器: {uri}")

    async with websockets.connect(uri) as ws:
        init_msg = json.loads(await ws.recv())
        if init_msg.get("type") != "connected":
            raise RuntimeError(f"Unexpected init message: {init_msg}")

        session_id = init_msg["session_id"]
        post_process_enabled = init_msg.get("post_process_enabled", False)
        print(f"会话 ID: {session_id}")
        print(f"后处理模块: {'已启用' if post_process_enabled else '未启用'}")
        print("=" * 60)

        await ws.send(json.dumps({
            "type": "config",
            "hotwords": hotwords,
            "city": city,
        }))

        print(f"流式识别结果:")

        pos = 0
        chunk_idx = 0
        total_encode_time = 0.0
        total_transfer_time = 0.0
        total_infer_time = 0.0
        total_decode_time = 0.0
        total_post_process_time = 0.0
        total_first_chunk_delay_time = 0.0

        t_start = time.time()
        while pos < wav16k.shape[0]:
            seg = wav16k[pos: pos + step]
            pos += seg.shape[0]
            chunk_idx += 1

            t_encode_start = time.time()
            audio_data = seg.tobytes()
            t_encode_end = time.time()

            t_send_start = time.time()
            await ws.send(audio_data)

            result = json.loads(await ws.recv())
            t_send_end = time.time()

            if result.get("type") == "result":
                print(f"  [块 {chunk_idx}] 语言={result['language']!r} 文本={result['text']!r}")
                if chunk_idx == 1:
                    total_first_chunk_delay_time = time.time() - t_start
                #print(f"          编码={result.get('decode_time', 0):.4f}s 推理={result.get('infer_time', 0):.4f}s")

                total_encode_time += t_encode_end - t_encode_start
                total_transfer_time += t_send_end - t_send_start
                total_infer_time += result.get("infer_time", 0)
                total_decode_time += result.get("decode_time", 0)

        await ws.send(json.dumps({"type": "finish"}))
        final_msg = json.loads(await ws.recv())
        total_post_process_time = final_msg.get("post_process_time", 0)

        print("\n" + "=" * 60)
        print(f"整句识别结果:")
        print(f"  语言: {final_msg.get('language', '')}")
        print(f"  原始文本: {final_msg.get('text', '')}")
        corrected_text = final_msg.get('corrected_text', final_msg.get('text', ''))
        if corrected_text != final_msg.get('text', ''):
            print(f"  纠错文本: {corrected_text}")
        if total_post_process_time > 0:
            print(f"  后处理耗时: {total_post_process_time:.4f}s")

        return {
            "language": final_msg.get("language", ""),
            "text": final_msg.get("text", ""),
            "corrected_text": corrected_text,
            "session_id": session_id,
            "stats": {
                "total_encode_time": total_encode_time,
                "total_transfer_time": total_transfer_time,
                "total_infer_time": total_infer_time,
                "total_decode_time": total_decode_time,
                "total_post_process_time": total_post_process_time,
                "total_first_chunk_delay_time": total_first_chunk_delay_time,
            }
        }


async def main():
    parser = argparse.ArgumentParser(description="Streaming VLLM Client (WebSocket + Opus)")
    parser.add_argument("--wav", type=str, required=True, help="Path to WAV file")
    parser.add_argument("--hotwords", type=str, default="", help="Hotwords, comma separated (e.g., 'word1,word2')")
    parser.add_argument("--city", type=str, default=None, help="City for post-processing POI correction")

    args = parser.parse_args()

    hotwords = [w.strip() for w in args.hotwords.split(",") if w.strip()]
    city = args.city

    audio_bytes = _read_audio_bytes(args.wav)
    wav, sr = _read_wav_from_bytes(audio_bytes)
    audio_duration = len(wav) / 16000

    step_ms = 500
    try:
        result = await test_streaming_vllm_ws(
            wav,
            step_ms=step_ms,
            hotwords=hotwords,
            city=city
        )

        t_total = (result['stats']['total_encode_time'] +
                   result['stats']['total_infer_time'] +
                   result['stats']['total_transfer_time'] +
                   result['stats']['total_decode_time'])
        t_transfer = result['stats']['total_transfer_time']
        rtf = t_total / audio_duration if audio_duration > 0 else 0

        print("\n" + "=" * 60)
        print(f"性能统计:")
        print(f"  音频时长: {audio_duration:.2f}s")
        print(f"  首字延时: {result['stats']['total_first_chunk_delay_time']:.4f}s")
        print(f"  编码时间: {result['stats']['total_encode_time']:.4f}s")
        print(f"  传输时间: {result['stats']['total_transfer_time']:.4f}s (网络延迟)")
        print(f"  推理时间: {result['stats']['total_infer_time']:.4f}s")
        print(f"  解码时间: {result['stats']['total_decode_time']:.4f}s")
        print(f"  后处理时间: {result['stats']['total_post_process_time']:.4f}s")
        print(f"  服务端处理时间: {result['stats']['total_infer_time'] + result['stats']['total_decode_time'] + result['stats']['total_post_process_time']:.4f}s")
        print(f"  RTF: {rtf:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
