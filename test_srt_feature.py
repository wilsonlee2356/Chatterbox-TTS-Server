# Quick functional test for utils.parse_srt and the /tts/srt timeline assembly logic.
import numpy as np
from utils import parse_srt

# --- Test 1: realistic SRT (CRLF, BOM, multi-line, tags, dot ms separator, out of order) ---
srt = """\ufeff1
00:00:01,000 --> 00:00:03,500
Hello, <i>welcome</i> to
the video.

2
00:00:10.250 --> 00:00:12.000
Second line with dot separator.

3
00:00:04,000 --> 00:00:06,000
Out of order entry.

4
00:00:12,500 --> 00:00:14,000


5
00:00:14,500 --> 00:00:16,000
<font color="#fff">Tagged</font> text only.
"""

entries = parse_srt(srt)
assert len(entries) == 4, f"expected 4 entries (empty-text one skipped), got {len(entries)}"
assert entries[0] == (1.0, 3.5, "Hello, welcome to the video."), entries[0]
assert entries[1] == (4.0, 6.0, "Out of order entry."), entries[1]  # sorted by start
assert entries[2] == (10.25, 12.0, "Second line with dot separator."), entries[2]
assert entries[3] == (14.5, 16.0, "Tagged text only."), entries[3]
print("Test 1 (parsing, tags, ordering, ms formats) OK")

# --- Test 2: error cases ---
for bad in ["", "   \n\n ", "no timestamps here\njust text"]:
    try:
        parse_srt(bad)
        raise AssertionError(f"should have raised ValueError for: {bad!r}")
    except ValueError:
        pass
print("Test 2 (ValueError on empty/invalid) OK")

# --- Test 3: timeline assembly (mirrors endpoint logic) ---
sr = 24000
subtitle_entries = [(0.5, 2.0, "a"), (2.5, 3.5, "b"), (3.5, 4.0, "c")]
# fake synthesized audio: segment "b" overflows its 1s slot into "c" (tests additive mixing)
placed = [
    (0.5, np.full(int(1.0 * sr), 0.5, dtype=np.float32)),   # 1s, fits
    (2.5, np.full(int(1.5 * sr), 0.4, dtype=np.float32)),   # 1.5s, overflows slot
    (3.5, np.full(int(0.4 * sr), 0.3, dtype=np.float32)),   # starts while b still playing
]

last_end = max(e for _, e, _ in subtitle_entries)
total = int(last_end * sr)
for st, seg in placed:
    total = max(total, int(st * sr) + len(seg))
total += sr

timeline = np.zeros(total, dtype=np.float32)
overflow = 0
for idx, (st, seg) in enumerate(placed):
    pos = int(st * sr)
    timeline[pos : pos + len(seg)] += seg
    if idx + 1 < len(placed) and st + len(seg) / sr > placed[idx + 1][0]:
        overflow += 1
peak = float(np.max(np.abs(timeline)))
if peak > 0.99:
    timeline *= 0.95 / peak

assert overflow == 1, overflow
# silence before first subtitle
assert np.all(timeline[: int(0.5 * sr)] == 0)
# first segment placed at its timestamp
assert np.isclose(timeline[int(0.75 * sr)], 0.5 * (0.95 / peak) if peak > 0.99 else 0.5)
# overlap zone [3.5s, 4.0s): b (0.4) + c (0.3) = 0.7 -> normalized since > 0.99? no, peak=0.7
assert peak <= 0.99
assert np.isclose(timeline[int(3.75 * sr)], 0.7, atol=1e-6), timeline[int(3.75 * sr)]
# gap between a-end (1.5s) and b-start (2.5s) is silence
assert np.all(timeline[int(1.6 * sr) : int(2.4 * sr)] == 0)
# total duration covers overflow + 1s tail
assert len(timeline) == int(4.0 * sr) + sr
print("Test 3 (timeline placement, silence gaps, additive overlap, normalization) OK")

# --- Test 4: lead/trail silence trimming before slot measurement (fit_to_slot fix) ---
from utils import trim_lead_trail_silence

tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(int(1.0 * sr)) / sr).astype(np.float32)
padded = np.concatenate([
    np.zeros(int(0.4 * sr), dtype=np.float32),   # 0.4s leading silence
    tone,                                         # 1.0s "speech"
    np.zeros(int(0.8 * sr), dtype=np.float32),   # 0.8s trailing silence
])
trimmed = trim_lead_trail_silence(padded, sr)
# Slot is 1.5s: untrimmed (2.2s) would trigger a stretch, trimmed (~1.1s w/ 50ms padding) must not.
assert len(padded) / sr > 1.5, f"precondition failed: {len(padded) / sr:.2f}s"
assert len(trimmed) / sr <= 1.5, f"trim failed: still {len(trimmed) / sr:.2f}s"
assert len(trimmed) / sr >= 1.0, f"over-trimmed: {len(trimmed) / sr:.2f}s"
print(f"Test 4 (silence trim: {len(padded)/sr:.2f}s -> {len(trimmed)/sr:.2f}s, avoids unnecessary stretch) OK")

# --- Test 5: WSOLA speed factor (echo-free replacement for librosa time_stretch) ---
from utils import apply_speed_factor_wsola

faster = apply_speed_factor_wsola(tone, 1.25)
assert faster.dtype == np.float32 and not np.isnan(faster).any()
ratio = len(faster) / len(tone)
assert 0.65 < ratio < 0.9, f"1.25x speedup gave ratio {ratio}"
slower = apply_speed_factor_wsola(tone, 0.8)
assert len(slower) > len(tone), "0.8x slowdown should lengthen audio"
same = apply_speed_factor_wsola(tone, 1.0)
assert len(same) == len(tone), "factor 1.0 must be a no-op"
invalid = apply_speed_factor_wsola(tone, -2.0)
assert len(invalid) == len(tone), "invalid factor must return original"
shorty = np.zeros(500, dtype=np.float32)
assert len(apply_speed_factor_wsola(shorty, 1.5)) == 500, "tiny clips must pass through"
print(f"Test 5 (WSOLA stretch 1.25x -> ratio {ratio:.2f}, no-op/invalid/short guards) OK")

print("\nALL TESTS PASSED")
