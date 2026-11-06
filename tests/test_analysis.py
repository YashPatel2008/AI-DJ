import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze-song.py"
SPEC = importlib.util.spec_from_file_location("analyze_song", MODULE_PATH)
analyze_song = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze_song)


class KeyAnalysisTests(unittest.TestCase):
    def test_parse_key_accepts_supported_key(self):
        self.assertEqual(analyze_song.parse_key("C# minor"), (1, "minor"))

    def test_parse_key_rejects_malformed_key(self):
        self.assertEqual(analyze_song.parse_key("H major"), (None, None))
        self.assertEqual(analyze_song.parse_key("C major extra"), (None, None))

    def test_same_key_has_maximum_compatibility(self):
        self.assertEqual(analyze_song.key_compatibility("A minor", "A minor"), 1.0)

    def test_key_semitone_difference_uses_shortest_direction(self):
        self.assertEqual(analyze_song.key_semitone_diff("C major", "B major"), -1)


class PhraseAnalysisTests(unittest.TestCase):
    def test_builds_eight_bar_phrases_from_downbeats(self):
        song = {
            "beat_times": np.arange(0.0, 64.0, 0.5),
            "boundaries": [0.0, 64.0],
            "labels": ["verse"],
        }

        phrases = analyze_song.build_phrase_groups(song, phrase_bars=(8,))

        self.assertTrue(phrases)
        self.assertEqual(phrases[0]["start"], 0.0)
        self.assertEqual(phrases[0]["end"], 16.0)
        self.assertEqual(phrases[0]["phrase_bars"], 8)


if __name__ == "__main__":
    unittest.main()
