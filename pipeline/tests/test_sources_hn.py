import json
import unittest
from pathlib import Path

from pipeline.sources import SourceError, hn

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


class FakeClient:
    """Serves recorded responses; ``pages`` maps a query to its list of pages."""

    def __init__(self, pages=None, default=None, fail=()):
        self.pages = pages or {}
        self.default = default
        self.fail = set(fail)
        self.calls = []
        self.requests_made = self.cache_hits = 0

    def get(self, url, params=None, ttl_days=None):
        self.calls.append(params)
        q = params["query"]
        if q in self.fail:
            raise SourceError("boom")
        pages = self.pages.get(q)
        if pages is None:
            return self.default or {"hits": []}
        idx = sum(1 for c in self.calls if c["query"] == q) - 1
        return pages[min(idx, len(pages) - 1)]


class HnTest(unittest.TestCase):
    def setUp(self):
        self.hits = {h["objectID"]: h for h in load("hn_search.json")["hits"]}

    def test_relevance_requires_fitness_term_near_link(self):
        vid = "AAAAAAAAAAA"
        self.assertTrue(hn.near_fitness_term(f"great deadlift cues https://youtu.be/{vid}", vid))
        self.assertFalse(hn.near_fitness_term(f"see https://youtu.be/{vid} for the samsung mobility story", vid))
        far = "squat " + "x" * 400 + f" https://youtu.be/{vid}"
        self.assertFalse(hn.near_fitness_term(far, vid))
        self.assertFalse(hn.near_fitness_term(f"exercism is great https://youtu.be/{vid}", vid))
        self.assertFalse(hn.near_fitness_term(f"muscle memory https://youtu.be/{vid}", vid))
        for off_topic in ["always exercise your right to silence", "that stretch of road", "Bit-squatting DNS hijacking",
                          "a documentary about drug rehab centers"]:
            self.assertFalse(hn.near_fitness_term(f"{off_topic} https://youtu.be/{vid}", vid), off_topic)

    def test_parse_hit(self):
        [m] = hn.parse_hit(self.hits["36829013"])  # Stuart McGill stretch, HTML-escaped link with t=
        self.assertEqual(m.video_id, "bcbuhePZZj0")
        self.assertEqual(m.author, "hn:idle_processor")
        self.assertEqual(m.kind, "hn")
        self.assertEqual(m.site, "news.ycombinator.com")
        self.assertEqual(m.url, "https://news.ycombinator.com/item?id=36829013")
        self.assertEqual(m.created_utc, self.hits["36829013"]["created_at_i"])
        self.assertIsNone(m.score)
        self.assertIn((459, ""), m.timestamps)
        self.assertTrue(m.needs_llm_check)
        self.assertNotIn("<a", m.text)

    def test_parse_hit_drops_off_topic(self):
        self.assertEqual(hn.parse_hit(self.hits["38023066"]), [])  # "mobility" only inside a URL
        self.assertEqual(hn.parse_hit({"author": None, "comment_text": "x"}), [])

    def test_collect_dedupes_and_tolerates_failures(self):
        page = load("hn_search.json")
        client = FakeClient(default=page, fail={"youtu.be squat"})
        mentions = hn.collect(client, keywords=["squat", "back pain"], max_pages=2)
        ids = sorted(m.video_id for m in mentions)
        # every query returns the same hits: each comment is counted once
        self.assertEqual(len(ids), len(set((m.url, m.video_id) for m in mentions)))
        self.assertIn("bcbuhePZZj0", ids)
        self.assertNotIn("Ln4rsxWq3WM", ids)
        self.assertEqual(len(client.calls), 4)

    def test_search_pages_by_date(self):
        def hit(i, t):
            return {"objectID": str(i), "created_at_i": t}

        full = {"hits": [hit(i, 1000 - i) for i in range(1000)]}
        last = {"hits": [hit(5000, 1)]}
        client = FakeClient(pages={"q": [full, last]})
        hits = hn.search(client, "q", max_pages=5)
        self.assertEqual(len(hits), 1001)
        self.assertEqual(client.calls[1]["numericFilters"], "created_at_i<1")
        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
