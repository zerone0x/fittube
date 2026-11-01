import json
import unittest
from pathlib import Path

from pipeline.sources import SourceError, html_to_text, tnation

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "https://t-nation.com"


def load(name):
    return json.loads((FIXTURES / name).read_text())


class FakeClient:
    """Serves recorded Discourse responses by URL."""

    def __init__(self, search_pages=None, fail_urls=()):
        self.topics = load("tnation_topics.json")
        self.search_pages = search_pages or {}
        self.fail_urls = set(fail_urls)
        self.calls = []
        self.requests_made = self.cache_hits = 0

    def get(self, url, params=None, ttl_days=None):
        self.calls.append((url, params))
        if url in self.fail_urls:
            raise SourceError("boom")
        if url == f"{BASE}/categories.json":
            return load("tnation_categories.json")
        if url == f"{BASE}/search.json":
            key = (params["q"], params["page"])
            return self.search_pages.get(key, load("tnation_search.json") if key[1] == 1 else {"posts": []})
        if url.startswith(f"{BASE}/t/"):
            return self.topics[url.split("/t/")[1].removesuffix(".json")]
        raise AssertionError(url)


def topic_post(key, number):
    topic = load("tnation_topics.json")[key]
    post = next(p for p in topic["post_stream"]["posts"] if p["post_number"] == number)
    return post, {"id": topic["id"], "slug": topic["slug"], "title": topic["title"], "category_id": topic["category_id"]}


class TNationTest(unittest.TestCase):
    def test_own_clip_heuristic(self):
        for text in ["Could someone tell me how my form looks?", "Here's my top set from today", "form check please",
                     "here's me pulling 500", "video of me squatting", "Check out my new PR",
                     "Pulled a new PR on deads this morning", "Yours truly picking the brain of"]:
            self.assertTrue(tnation.is_own_clip(text), text)
        for text in ["Here's a good video on hip hinging", "This guy explains the squat really well", ""]:
            self.assertFalse(tnation.is_own_clip(text), text)

    def test_html_to_text_drops_quotes_keeps_onebox_link(self):
        post, _ = topic_post("294151/4", 4)
        text, links = html_to_text(post["cooked"])
        self.assertEqual(text, "I would focus on making it a loaded hip hinge.\nHere’s a good video")
        self.assertEqual(links, ["https://www.youtube.com/watch?v=jEy_czb3RKA"])
        quoted, _ = topic_post("294806/1", 2)
        text, _ = html_to_text(quoted["cooked"])
        self.assertNotIn("FrankPP", text)
        self.assertNotIn("My primary goal", text)

    def test_parse_post(self):
        post, topic = topic_post("294151/4", 4)
        [m] = tnation.parse_post(post, topic, like_count=1)
        self.assertEqual(m.video_id, "jEy_czb3RKA")
        self.assertEqual(m.author, "tnation:t3hpwnisher")
        self.assertEqual((m.kind, m.site), ("forum", "t-nation.com"))
        self.assertEqual(m.url, "https://t-nation.com/t/should-rdl-s-give-you-a-lower-back-pump/294151/4")
        self.assertEqual(m.score, 1)
        self.assertEqual(m.created_utc, 1780798755)
        self.assertTrue(m.needs_llm_check)

    def test_parse_post_filters(self):
        own, topic = topic_post("294806/1", 1)  # "how my form looks" + own clip
        self.assertEqual(tnation.parse_post(own, topic), [])
        offtopic, topic = topic_post("294191/29", 29)  # soccer video in the Main Forum lounge
        self.assertEqual(len(tnation.parse_post(offtopic, topic, trusted=True)), 1)
        self.assertEqual(tnation.parse_post(offtopic, topic, trusted=False), [])
        self.assertEqual(tnation.parse_post({**offtopic, "hidden": True}, topic), [])
        good, topic = topic_post("294151/4", 4)
        self.assertEqual(tnation.parse_post(good, {**topic, "title": "(Un)official 2026 T-ransformation Challenge"}), [])
        self.assertEqual(tnation.parse_post(good, {**topic, "title": "Rising from the ashes - training log"}), [])

    def test_collect(self):
        client = FakeClient()
        mentions = tnation.collect(client, max_pages=3)
        self.assertEqual([m.video_id for m in mentions], ["jEy_czb3RKA"])
        fetched = [u for u, _ in client.calls if "/t/" in u]
        # training-log post (category 5) is never fetched
        self.assertEqual(fetched, [f"{BASE}/t/294151/4.json", f"{BASE}/t/294806/1.json", f"{BASE}/t/294191/29.json"])

    def test_collect_survives_failures(self):
        self.assertEqual(tnation.collect(FakeClient(fail_urls={f"{BASE}/categories.json"})), [])
        client = FakeClient(fail_urls={f"{BASE}/t/294151/4.json"})
        self.assertEqual(tnation.collect(client), [])

    def test_search_pages_with_before_cursor(self):
        def page(ids, day, more=True):
            return {"posts": [{"id": i, "topic_id": 1, "post_number": i, "created_at": f"{day}T10:00:00Z"} for i in ids],
                    "topics": [{"id": 1, "category_id": 10}],
                    "grouped_search_result": {"more_full_page_results": more}}

        q = tnation.config.TNATION_QUERY
        pages = {(q, n): page([n], "2026-10-%02d" % (30 - n)) for n in range(1, 11)}
        pages[(f"{q} before:2026-10-21", 1)] = page([10, 11], "2026-10-19", more=False)  # overlap is de-duplicated
        client = FakeClient(search_pages=pages)
        hits = tnation.search_posts(client, max_pages=30, drop_category_ids=set())
        self.assertEqual([p["id"] for p, _ in hits], list(range(1, 12)))
        self.assertEqual(len(client.calls), 11)
        # the page cap is honoured across cursors
        self.assertEqual(len(tnation.search_posts(FakeClient(search_pages=pages), 4, set())), 4)


if __name__ == "__main__":
    unittest.main()
