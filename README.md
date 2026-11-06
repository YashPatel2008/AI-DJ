<div align="center">

# AIDJ

### A music-aware DJ that finds the moment two tracks belong together.

Analyze rhythm, harmony, energy, song sections, and vocal activity to rank
transition points and render an automatically crossfaded mix.

</div>

---

## What it does

AIDJ is a command-line audio analysis and transition-rendering experiment. Give
it two audio files and it:

1. **Analyzes each track** for tempo, beats, musical key, energy, and structural
   boundaries.
2. **Separates stems** into vocals, drums, bass, and other using Demucs.
3. **Finds phrase-aware transition candidates**, weighing tempo and key
   compatibility, energy similarity, and vocal overlap.
4. **Renders a mix** with optional pitch and tempo adjustment and an equal-power
   crossfade.
5. **Caches analysis locally** so repeat runs can skip expensive work.

This is an evolving personal project. Automated music analysis can make mistakes;
listen to the rendered mix and review the suggested transition before using it.

## Quick start

### Requirements

- Python 3.12
- FFmpeg, required by the audio-analysis and stem-separation dependencies
- Rubber Band command-line tools, used for pitch shifting
- Several gigabytes of free disk space for the Demucs model and analysis cache

Install the system tools:

```bash
# macOS
brew install ffmpeg rubberband

# Debian / Ubuntu
sudo apt-get install ffmpeg rubberband-cli
```

Install the Python dependencies into an available Python installation:

```bash
python3 -m pip install -r requirements.txt
```

### Run a transition

Place two tracks you have permission to use on your computer, then run:

```bash
python3 analyze-song.py \
  --song-a "/path/to/track-a.mp3" \
  --song-b "/path/to/track-b.mp3"
```

The first run may take a while: Demucs downloads its pretrained model and AIDJ
analyzes both full tracks. The default mix is written to `mix_output.wav`.

Useful options:

```bash
python3 analyze-song.py --help
python3 analyze-song.py \
  --song-a "/path/to/track-a.wav" \
  --song-b "/path/to/track-b.mp3" \
  --output "output/transition.wav" \
  --crossfade-seconds 12 \
  --cache-dir cache \
  --stems-dir separated_output
```

The script also writes separated stems under `separated_output/`. Both the mix
and stems are generated locally and are excluded from Git.

## Project layout

| Path | Purpose |
| --- | --- |
| `analyze-song.py` | Audio analysis, transition ranking, and mix rendering |
| `requirements.txt` | Pinned Python dependencies |
| `estimations/` | Small example MSAF structure-analysis results |
| `cache/` | Local, reusable analysis cache; cache data is ignored by Git because it can be very large |
| `songs/` | Put your own input tracks here; audio is ignored by Git |
| `separated_output/` | Locally generated Demucs stems; ignored by Git |

No copyrighted songs, stems, or rendered mixes are included. Keep your local
cache if it is useful; it will be reused when the source track still matches.
If a cache entry is missing or stale, AIDJ analyzes the track again.

## How transition ranking works

Candidate transitions are scored from tempo similarity, estimated key
compatibility, energy similarity, and vocal conflict. AIDJ groups beats into
phrases within detected song sections, ranks promising exit/entry combinations,
and uses the top match for rendering. The analysis is heuristic: it is a
starting point for experimentation, not a substitute for listening.

## Development

The project currently centers on one executable Python file, so you can inspect
or run the complete workflow without navigating a framework or service stack.
The installed packages are heavyweight audio/ML dependencies; consult
`requirements.txt` for the tested versions.

## License and audio rights

Only use music you own or are authorized to process. AIDJ does not grant rights
to any input track or generated mix. Add a project license before redistributing
the code if you want to specify reuse terms.
