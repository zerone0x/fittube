"""Data-quality check (pipeline/check_data.py)."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import check_data


def good_video(vid="AAAAAAAAAAA", **kw):
    v = {"id": vid, "title": "How to squat", "summary": "Shows how to squat with a neutral back.",
         "comments": [{"text": "great https://www.youtube.com/watch?v=AAAAAAAAAAA&t=30", "score": 3}],
         "sources": [{"kind": "comment", "subreddit": "Fitness"}],
         "source_stats": {"reddit": {"recommenders": 2, "sum_score": 4.0}}}
    v.update(kw)
    return v


class CheckDataTest(unittest.TestCase):
    def problems(self, **kw):
        return check_data.check({"videos": [good_video(**kw)]})

    def test_clean_data_passes(self):
        self.assertEqual(check_data.check({"videos": [good_video(), good_video("BBBBBBBBBBB", summary=None)]}),
                         ["BBBBBBBBBBB: summary empty: None"])
        no_summary = good_video("BBBBBBBBBBB")
        del no_summary["summary"]  # optional
        self.assertEqual(check_data.check({"videos": [good_video(), no_summary]}), [])

    def test_failures(self):
        cases = {
            "summary markup": {"summary": "Shows how to squat.}</parameter>"},
            "summary provenance": {"summary": "Shows how to squat, recommended on r/Fitness."},
            "summary meta": {"summary": "Shows how to squat; details beyond the title are not given."},
            "summary truncated": {"summary": "Shows how to squat with a neutral b…"},
            "summary reception": {"summary": "Shows how to squat, recommended as a brief, to-the-point guide."},
            "internal field 'llm_rejected'": {"llm_rejected": {"recommenders": 1}},
            "internal field 'llm_note'": {"llm_note": "kept-by-content"},
            "source_stats.forum without a forum source": {
                "source_stats": {"reddit": {"recommenders": 2}, "forum": {"recommenders": 1}}},
            "glued timestamp": {"comments": [{"text": "see watch?v=AAAAAAAAAAAt=268", "score": 1}]},
        }
        for want, kw in cases.items():
            problems = self.problems(**kw)
            self.assertTrue(any(want in p for p in problems), (want, problems))
        for summary in ("Recommended by commenters as a squat tutorial.", "Squat tutorial, per community comments.",
                        "Squat tutorial; the exact method is unclear from the title.",
                        'Video titled "Dog of Wisdom" from channel Joe.'):
            self.assertTrue(self.problems(summary=summary), summary)
        self.assertEqual(self.problems(summary="Shows squats at home, recommended for people with limited equipment."),
                         [])
        self.assertEqual(check_data.check({"videos": []}), ["no videos"])
        self.assertIn("BBBBBBBBBBB: duplicate id",
                      check_data.check({"videos": [good_video("BBBBBBBBBBB"), good_video("BBBBBBBBBBB")]}))

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "videos.json"
            path.write_text(json.dumps({"videos": [good_video()]}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(check_data.main([str(path)]), 0)
            self.assertIn("OK (1 videos, 1 summaries)", out.getvalue())
            path.write_text(json.dumps({"videos": [good_video(llm_note="x")]}))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(check_data.main([str(path)]), 1)
                self.assertEqual(check_data.main([str(Path(tmp) / "missing.json")]), 2)


if __name__ == "__main__":
    unittest.main()
