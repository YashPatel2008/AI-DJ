import argparse
import librosa
import msaf
import numpy as np
import demucs
from demucs.api import Separator
import soundfile as sf
import os
import pickle
import pyrubberband as pyrb
from pathlib import Path
from scipy.signal import butter, lfilter
 
# Key Profiles for major and minor keys
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
 
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
 
 
def sample_curve(time, times, values):
    if len(times) == 0:
        return 0.0
    index = int(np.argmin(np.abs(times - time)))
    return float(values[index])
 
 
def window_average(times, values, start, end):
    if len(times) == 0 or start >= end:
        return 0.0
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        return float(values[np.argmin(np.abs(times - (start + end) / 2))])
    return float(np.mean(values[mask]))
 
 
def make_mono(audio):
    audio = np.asarray(audio)
    if audio.ndim == 2:
        if audio.shape[0] <= audio.shape[1]:
            audio = np.mean(audio, axis=0)
        else:
            audio = np.mean(audio, axis=1)
    return audio
 
 
def butter_filter(data, cutoff, fs, btype='low', order=2):
    nyq = 0.5 * fs
    normal_cutoff = np.clip(cutoff / nyq, 0.001, 0.999)
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, data)
 
 
def detect_key(chroma):
    chroma_avg = np.mean(chroma, axis=1)
    best_score = -1
    best_key = None
 
    for i in range(12):
        major_rotated = np.roll(MAJOR_PROFILE, i)
        minor_rotated = np.roll(MINOR_PROFILE, i)
 
        major_score = np.corrcoef(chroma_avg, major_rotated)[0, 1]
        minor_score = np.corrcoef(chroma_avg, minor_rotated)[0, 1]
 
        if major_score > best_score:
            best_score = major_score
            best_key = f"{NOTE_NAMES[i]} major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{NOTE_NAMES[i]} minor"
 
    return best_key, float(best_score)
 
 
def vocal_density_at(song, center_time, window=2.0):
    vocal = song.get("vocal_activity")
    if vocal is None:
        return 0.0
    return window_average(vocal["times"], vocal["mask"].astype(float), center_time - window / 2, center_time + window / 2)
 
 
def get_exit_candidates(song):
    energy_times = song.get("energy_times")
    energy_curve = song.get("energy_curve")[0]
    phrase_groups = song.get("phrase_groups", [])
    boundaries = song.get("boundaries", [])
 
    candidates = []
    for phrase in phrase_groups:
        exit_time = phrase["end"]
        if exit_time < 4.0 or exit_time > song.get("duration", 0.0) - 4.0:
            continue
 
        energy_value = sample_curve(exit_time, energy_times, energy_curve)
        vocal_density = vocal_density_at(song, exit_time, window=2.0)
        repeat_count = song["labels"].count(phrase["label"])
        section_end = boundaries[phrase["section_index"] + 1] if phrase["section_index"] + 1 < len(boundaries) else exit_time
        stability_score = 1.0 if abs(exit_time - section_end) < 0.5 else 0.88
 
        score = energy_value * (1.0 + repeat_count * 0.12) * stability_score * (1.0 - vocal_density * 0.55)
        candidates.append({
            "time": float(exit_time),
            "label": phrase["label"],
            "energy": float(energy_value),
            "vocal_density": float(vocal_density),
            "repeat_count": repeat_count,
            "score": float(score),
            "phrase_duration": phrase["duration"],
            "phrase": phrase,
        })
 
    return sorted(candidates, key=lambda c: c["score"], reverse=True)
 
 
def get_entry_candidates(song, min_remaining=30.0):
    energy_times = song.get("energy_times")
    energy_curve = song.get("energy_curve")[0]
    phrase_groups = song.get("phrase_groups", [])
    duration = song.get("duration", 0.0)
 
    candidates = []
    for phrase in phrase_groups:
        entry_time = phrase["start"]
        if entry_time < 2.0 or entry_time > duration - min_remaining:
            continue
 
        energy_value = sample_curve(entry_time, energy_times, energy_curve)
        vocal_density = vocal_density_at(song, entry_time + 1.0, window=2.0)
        repeat_count = song["labels"].count(phrase["label"])
        score = energy_value * (1.0 + repeat_count * 0.12) * (1.0 - vocal_density * 0.4)
        candidates.append({
            "time": float(entry_time),
            "label": phrase["label"],
            "energy": float(energy_value),
            "vocal_density": float(vocal_density),
            "repeat_count": repeat_count,
            "score": float(score),
            "phrase_duration": phrase["duration"],
            "phrase": phrase,
        })
 
    return sorted(candidates, key=lambda c: c["score"], reverse=True)
 
 
def save_analysis(song_analysis, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(song_analysis, f)
 
 
REQUIRED_ANALYSIS_KEYS = {
    "tempo",
    "beat_times",
    "energy_curve",
    "energy_times",
    "key",
    "boundaries",
    "labels",
    "phrase_groups",
    "duration",
    "sample_rate",
    "vocal_activity",
}
 
 
def load_analysis(cache_path):
    with open(cache_path, 'rb') as f:
        return pickle.load(f)
 
 
def is_analysis_valid(analysis):
    if analysis is None or not isinstance(analysis, dict):
        return False
    return REQUIRED_ANALYSIS_KEYS.issubset(set(analysis.keys()))
 
 
def normalize_song_path(song_path):
    return os.path.abspath(os.path.normpath(song_path))
 
 
def get_song_file_signature(song_path):
    song_path = normalize_song_path(song_path)
    if not os.path.exists(song_path):
        return song_path, None, None
    stat = os.stat(song_path)
    return song_path, int(stat.st_mtime_ns), int(stat.st_size)
 
 
def is_analysis_for_current_song(analysis, song_path):
    if not is_analysis_valid(analysis):
        return False
 
    expected_path, expected_mtime, expected_size = get_song_file_signature(song_path)
    stored_path = analysis.get("file_path")
    if stored_path is not None and normalize_song_path(stored_path) != expected_path:
        return False
    if stored_path is None:
        return False
 
    actual_mtime = analysis.get("source_mtime")
    actual_size = analysis.get("source_size")
    if expected_mtime is not None and actual_mtime is not None:
        if actual_mtime != expected_mtime:
            return False
    if expected_size is not None and actual_size is not None:
        if actual_size != expected_size:
            return False
    return True
 
 
def get_or_analyze(song_path, cache_path):
    song_path = normalize_song_path(song_path)
    cache_path = normalize_song_path(cache_path)
 
    if os.path.exists(cache_path):
        analysis = load_analysis(cache_path)
        if not is_analysis_for_current_song(analysis, song_path):
            print(f"Cached analysis for {song_path} is outdated or belongs to a different song; reanalyzing...")
            analysis = analyze_Song(song_path)
            analysis["file_path"] = song_path
            _, source_mtime, source_size = get_song_file_signature(song_path)
            analysis["source_mtime"] = source_mtime
            analysis["source_size"] = source_size
            save_analysis(analysis, cache_path)
            return analysis
        print(f"Loading cached analysis for {song_path}...")
        analysis["file_path"] = song_path
        _, source_mtime, source_size = get_song_file_signature(song_path)
        analysis["source_mtime"] = source_mtime
        analysis["source_size"] = source_size
        return analysis
    print(f"Analyzing {song_path} (no cache found)...")
    result = analyze_Song(song_path)
    result["file_path"] = song_path
    _, source_mtime, source_size = get_song_file_signature(song_path)
    result["source_mtime"] = source_mtime
    result["source_size"] = source_size
    save_analysis(result, cache_path)
    return result
 
 
def save_stems(stems, output_dir, sample_rate=44100):
    os.makedirs(output_dir, exist_ok=True)
    for stem_name, tensor in stems.items():
        audio = tensor.numpy().T
        filepath = os.path.join(output_dir, f"{stem_name}.wav")
        sf.write(filepath, audio, sample_rate)
 
 
def separate_stems(song_path):
    separator = Separator(model="htdemucs")
    origin, separated = separator.separate_audio_file(song_path)
    return separated
 
 
def load_song_wave(song):
    wave, sr = librosa.load(song["file_path"], sr=song["sample_rate"])
    return make_mono(wave), sr
 
 
def load_stem_wave(song, stem_name):
    stem = song.get("stems", {}).get(stem_name)
    if stem is None:
        return None
    return make_mono(stem.numpy() if hasattr(stem, "numpy") else np.asarray(stem))
 
 
def render_transition_mix(song_a, song_b, pair, output_path, crossfade_seconds=15.0):
    wave_a, sr_a = load_song_wave(song_a)
    wave_b, sr_b = load_song_wave(song_b)
    if sr_a != sr_b:
        wave_b = librosa.resample(y=wave_b, orig_sr=sr_b, target_sr=sr_a)
        sr_b = sr_a
    sr = sr_a
 
    # FIXED: was using the key_semitone_diff *function* by mistake instead of
    # the actual number stored in pair["key_semitone_diff"]
    if pair.get("key_semitone_diff"):
        wave_b = pyrb.pitch_shift(wave_b, sr, n_steps=pair["key_semitone_diff"])
 
    tempo_diff = song_b["tempo"] - song_a["tempo"]
    if abs(tempo_diff) > 2.0:
        rate = song_a["tempo"] / song_b["tempo"]
        wave_b = librosa.effects.time_stretch(y=wave_b, rate=rate)
        entry_sample = int(round(pair["entry_time"] * sr * rate))
    else:
        entry_sample = int(round(pair["entry_time"] * sr))
 
    exit_sample = int(round(pair["exit_time"] * sr))
    fade_len = int(round(crossfade_seconds * sr))
 
    # FIXED: was stepping back only fade_len // 2, which silently halved
    # the actual overlap duration compared to what crossfade_seconds requested
    fade_start = max(0, exit_sample - fade_len)   # A starts fading out here
    a_lead_in = wave_a[:fade_start]                 # A plays alone, untouched
 
    a_fade_segment = wave_a[fade_start:exit_sample]  # A's fading-out portion
    b_fade_segment = wave_b[entry_sample: entry_sample + len(a_fade_segment)]
 
    # Match lengths in case B runs short
    cross_len = min(len(a_fade_segment), len(b_fade_segment))
    a_fade_segment = a_fade_segment[:cross_len]
    b_fade_segment = b_fade_segment[:cross_len]
 
    if pair["energy_score"] < 0.7 and len(b_fade_segment) > 0:
        b_fade_segment = butter_filter(b_fade_segment, cutoff=1200.0, fs=sr, btype='low')
 
    # Equal-power crossfade curve (keeps combined loudness fuller through the blend
    # instead of dipping in the middle like a straight linear fade would)
    t = np.linspace(0.0, 1.0, cross_len)
    a_curve = np.cos(t * np.pi / 2)
    b_curve = np.sin(t * np.pi / 2)
    crossfade_zone = a_fade_segment * a_curve + b_fade_segment * b_curve
 
    b_tail = wave_b[entry_sample + cross_len:]  # B plays alone after the blend
 
    mix = np.concatenate([a_lead_in, crossfade_zone, b_tail]).astype(np.float32)
    mix = np.clip(mix, -1.0, 1.0)
 
    sf.write(output_path, mix, sr)
    return output_path
 
 
def parse_key(key_name):
    if not key_name or not isinstance(key_name, str):
        return None, None
    parts = key_name.strip().split()
    if len(parts) != 2:
        return None, None
    tonic, mode = parts
    if tonic not in NOTE_NAMES or mode not in {"major", "minor"}:
        return None, None
    return NOTE_NAMES.index(tonic), mode
 
 
def key_compatibility(key_a, key_b):
    tonic_a, mode_a = parse_key(key_a)
    tonic_b, mode_b = parse_key(key_b)
    if tonic_a is None or tonic_b is None:
        return 0.6
 
    diff = min((tonic_a - tonic_b) % 12, (tonic_b - tonic_a) % 12)
    if tonic_a == tonic_b:
        return 1.0 if mode_a == mode_b else 0.95
    if mode_a != mode_b and diff in {3, 9}:
        return 0.92
    if diff in {1, 11}:
        return 0.88
    if diff in {5, 7}:
        return 0.8
    if diff in {2, 10}:
        return 0.68
    return 0.45
 
 
def build_phrase_groups(song, phrase_bars=(8, 16), min_phrase_duration=4.0):
    beat_times = song.get("beat_times", np.array([]))
    boundaries = song.get("boundaries", [])
    labels = song.get("labels", [])
    if len(beat_times) == 0 or len(boundaries) < 2:
        return []
 
    downbeats = beat_times[::4]
    phrase_groups = []
    tolerance = 0.1
 
    for section_idx in range(len(boundaries) - 1):
        section_start = boundaries[section_idx]
        section_end = boundaries[section_idx + 1]
        section_downbeats = [t for t in downbeats if t >= section_start - tolerance and t <= section_end + tolerance]
        if len(section_downbeats) < 2:
            continue
 
        for bars in phrase_bars:
            for start_idx in range(0, len(section_downbeats) - bars + 1):
                start_time = section_downbeats[start_idx]
                end_idx = start_idx + bars
                end_time = section_downbeats[end_idx] if end_idx < len(section_downbeats) else section_end
                if end_time > section_end + tolerance:
                    continue
                duration = end_time - start_time
                if duration < min_phrase_duration:
                    continue
                phrase_groups.append({
                    "start": float(start_time),
                    "end": float(end_time),
                    "label": labels[section_idx],
                    "section_index": section_idx,
                    "duration": float(duration),
                    "phrase_bars": bars,
                    "phrase_type": f"{bars}-bar",
                    "start_index": start_idx,
                    "end_index": end_idx,
                })
 
    phrase_groups.sort(key=lambda x: (x["start"], -x["phrase_bars"]))
    return phrase_groups
 
 
def get_vocal_activity(song, stem_name="vocals", frame_length=2048, hop_length=512):
    stems = song.get("stems", {})
    stem = stems.get(stem_name)
    if stem is None:
        raise ValueError(f"Vocal stem '{stem_name}' not available in analysis")
 
    if hasattr(stem, "numpy"):
        audio = stem.numpy()
    else:
        audio = np.asarray(stem)
 
    if audio.ndim == 2:
        if audio.shape[0] <= audio.shape[1]:
            audio = np.mean(audio, axis=0)
        else:
            audio = np.mean(audio, axis=1)
 
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(energy)), sr=song["sample_rate"], hop_length=hop_length)
 
    median_energy = np.median(energy)
    dynamic_range = float(np.max(energy) - np.min(energy))
    threshold = max(median_energy + dynamic_range * 0.15, median_energy * 1.2, np.percentile(energy, 70) * 0.6)
 
    mask = energy >= threshold
    smoothed = np.convolve(mask.astype(float), np.ones(3) / 3, mode="same") > 0.5
    activity = {
        "times": times,
        "energy": energy,
        "mask": smoothed,
        "threshold": float(threshold),
    }
    song["vocal_activity"] = activity
    return activity
 
 
def key_semitone_diff(key_a, key_b):
    tonic_a, _ = parse_key(key_a)
    tonic_b, _ = parse_key(key_b)
    if tonic_a is None or tonic_b is None:
        return None
    diff = (tonic_b - tonic_a) % 12
    if diff > 6:
        diff -= 12
    return int(diff)
 
 
def compatibility_score(exit_candidate, entry_candidate, song_a, song_b):
    tempo_diff = abs(song_a["tempo"] - song_b["tempo"])
    tempo_score = max(0.0, 1.0 - tempo_diff / 10.0)
    key_score = key_compatibility(song_a["key"], song_b["key"])
    energy_score = 1.0 - min(1.0, abs(exit_candidate["energy"] - entry_candidate["energy"]) / max(exit_candidate["energy"], entry_candidate["energy"], 1e-4))
    vocal_conflict = 1.0 - min(1.0, exit_candidate["vocal_density"] + entry_candidate["vocal_density"])
    score = tempo_score * 0.33 + key_score * 0.33 + energy_score * 0.2 + vocal_conflict * 0.14
 
    return {
        "exit_time": exit_candidate["time"],
        "entry_time": entry_candidate["time"],
        "tempo_score": float(tempo_score),
        "key_score": float(key_score),
        "energy_score": float(energy_score),
        "vocal_conflict": float(vocal_conflict),
        "key_semitone_diff": key_semitone_diff(song_a["key"], song_b["key"]),
        "score": float(score),
        "exit": exit_candidate,
        "entry": entry_candidate,
    }
 
 
def rank_transition_pairs(song_a, song_b, exits, entries, top_n=5):
    pairs = []
    for exit_candidate in exits[:10]:
        for entry_candidate in entries[:10]:
            pairs.append(compatibility_score(exit_candidate, entry_candidate, song_a, song_b))
    return sorted(pairs, key=lambda p: p["score"], reverse=True)[:top_n]
 
 
def build_transition_plan(pair, song_a, song_b):
    exit_phrase = pair["exit"]["phrase"]
    entry_phrase = pair["entry"]["phrase"]
    semitone_diff = pair["key_semitone_diff"]
    tempo_diff = song_b["tempo"] - song_a["tempo"]
 
    actions = [
        f"Exit song A at {pair['exit_time']:.2f}s (phrase {exit_phrase['phrase_bars']}-bar, label={exit_phrase['label']}).",
        f"Enter song B at {pair['entry_time']:.2f}s (phrase {entry_phrase['phrase_bars']}-bar, label={entry_phrase['label']}).",
    ]
 
    if abs(semitone_diff or 0) > 0:
        actions.append(
            f"Pitch-adjust song B by {semitone_diff:+d} semitone(s) for a tighter harmonic match."
        )
    elif pair["key_score"] > 0.9:
        actions.append("Keep both songs in their detected keys; the harmonic match is strong.")
 
    if pair["vocal_conflict"] < 0.5:
        actions.append(
            "Drop or duck song A vocals for 2 bars before the transition, then bring song B vocals in smoothly."
        )
    else:
        actions.append(
            "Use a vocal swap: cut song A vocals out for 2 beats, then introduce song B vocals with a filtered pad underneath."
        )
 
    if pair["energy_score"] < 0.7:
        actions.append(
            "Use a gentle filter sweep and delay + reverb to ease the energy change between songs."
        )
    else:
        actions.append(
            "Keep the energy momentum with a percussion build on song A and a strong downbeat entry for song B."
        )
 
    if abs(tempo_diff) > 3:
        actions.append(
            f"Apply tempo warping: stretch song B by {tempo_diff:+.1f} BPM to align beats before the transition."
        )
    else:
        actions.append("Keep both tracks at the same BPM; no major tempo warp needed.")
 
    if pair["exit"]["vocal_density"] > 0.3:
        actions.append(
            "Create a short vocal chop / gated repeat from song A's last phrase to lead into song B."
        )
 
    plan = {
        "exit_time": pair["exit_time"],
        "entry_time": pair["entry_time"],
        "score": pair["score"],
        "steps": actions,
        "pair": pair,
    }
    return plan
 
 
def analyze_Song(song_path):
    waveform, sample_rate = librosa.load(song_path, sr=None)
    duration = librosa.get_duration(y=waveform, sr=sample_rate)
    tempo = float(librosa.beat.tempo(y=waveform, sr=sample_rate)[0])
    _, beat_frames = librosa.beat.beat_track(y=waveform, sr=sample_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    energy_curve = librosa.feature.rms(y=waveform)
    energy_times = librosa.frames_to_time(np.arange(energy_curve.shape[1]), sr=sample_rate)
    key_curve = librosa.feature.chroma_cqt(y=waveform, sr=sample_rate)
    key, key_confidence = detect_key(key_curve)
    stems = separate_stems(song_path)
 
    boundaries, labels = msaf.process(song_path, boundaries_id="foote", labels_id="fmc2d")
    song_analysis = {
        "tempo": tempo,
        "beat_times": beat_times,
        "downbeats": beat_times[::4],
        "energy_curve": energy_curve,
        "energy_times": energy_times,
        "key_curve": key_curve,
        "key": key,
        "key_confidence": key_confidence,
        "boundaries": boundaries,
        "labels": labels,
        "stems": stems,
        "duration": duration,
        "sample_rate": sample_rate,
    }
    song_analysis["phrase_groups"] = build_phrase_groups(song_analysis, phrase_bars=(8, 16))
    get_vocal_activity(song_analysis, stem_name="vocals")
    return song_analysis
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Analyze two songs and render a harmonically aware DJ transition."
    )
    parser.add_argument("--song-a", required=True, type=Path, help="Track to mix out of")
    parser.add_argument("--song-b", required=True, type=Path, help="Track to mix into")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mix_output.wav"),
        help="Rendered mix path (default: mix_output.wav)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Analysis cache directory (default: cache/)",
    )
    parser.add_argument(
        "--stems-dir",
        type=Path,
        default=Path("separated_output"),
        help="Directory for separated stems (default: separated_output/)",
    )
    parser.add_argument(
        "--crossfade-seconds",
        type=float,
        default=15.0,
        help="Transition duration in seconds (default: 15)",
    )
    args = parser.parse_args()

    for song_path in (args.song_a, args.song_b):
        if not song_path.is_file():
            parser.error(f"audio file not found: {song_path}")
    if args.crossfade_seconds <= 0:
        parser.error("--crossfade-seconds must be greater than zero")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    song_a = get_or_analyze(
        str(args.song_a), str(args.cache_dir / f"{args.song_a.stem}.pkl")
    )
    save_stems(song_a["stems"], str(args.stems_dir / args.song_a.stem))

    song_b = get_or_analyze(
        str(args.song_b), str(args.cache_dir / f"{args.song_b.stem}.pkl")
    )
    save_stems(song_b["stems"], str(args.stems_dir / args.song_b.stem))

    print("\n--- SONG A: " + os.path.basename(song_a["file_path"]) + " ---")
    print(f"Tempo: {song_a['tempo']:.1f} BPM")
    print(f"Key: {song_a['key']} (confidence: {song_a['key_confidence']:.3f})")
    print(f"Phrase groups: {len(song_a.get('phrase_groups', []))}")
    print(f"Vocal activity frames: {len(song_a.get('vocal_activity', {}).get('times', []))}")
 
    print("\n--- SONG B: " + os.path.basename(song_b["file_path"]) + " ---")
    print(f"Tempo: {song_b['tempo']:.1f} BPM")
    print(f"Key: {song_b['key']} (confidence: {song_b['key_confidence']:.3f})")
    print(f"Phrase groups: {len(song_b.get('phrase_groups', []))}")
    print(f"Vocal activity frames: {len(song_b.get('vocal_activity', {}).get('times', []))}")
 
    exits_a = get_exit_candidates(song_a)
    entries_b = get_entry_candidates(song_b)
 
    print("\nTop exit candidates in " + os.path.basename(song_a["file_path"]) + ":")
    for c in exits_a[:5]:
        print(c)
 
    print("\nTop entry candidates in " + os.path.basename(song_b["file_path"]) + ":")
    for c in entries_b[:5]:
        print(c)
 
    pairs = rank_transition_pairs(song_a, song_b, exits_a, entries_b, top_n=5)
    print("\nTop transition pairs from " + os.path.basename(song_a["file_path"]) + " -> " + os.path.basename(song_b["file_path"]) + ":")
    for p in pairs:
        print({
            "exit_time": p["exit_time"],
            "entry_time": p["entry_time"],
            "score": round(p["score"], 3),
            "tempo_score": round(p["tempo_score"], 3),
            "key_score": round(p["key_score"], 3),
            "energy_score": round(p["energy_score"], 3),
            "vocal_conflict": round(p["vocal_conflict"], 3),
            "key_semitone_diff": p.get("key_semitone_diff"),
        })
 
    plan = build_transition_plan(pairs[0], song_a, song_b) if pairs else None
    if plan:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rendered_path = render_transition_mix(
            song_a,
            song_b,
            pairs[0],
            str(args.output),
            crossfade_seconds=args.crossfade_seconds,
        )
        print(f"\nRendered mix written to: {rendered_path}")


if __name__ == "__main__":
    main()
 