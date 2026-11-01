"""Tests for the Reddit wiki/post sources, using recorded Arctic Shift responses."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import build, config, fetch
from pipeline.reddit import posts, wiki

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class RecordedSession:
    """Serves recorded responses by endpoint; records the calls."""

    def __init__(self, routes, status=200):
        self.routes = routes
        self.status = status
        self.calls = []
        self.headers = {}

    def get(self, url, params, timeout):
        self.calls.append((url, dict(params)))
        if self.status != 200:
            return FakeResponse(self.status, {"data": None, "error": "Timeout. Maybe slow down a bit"})
        return FakeResponse(200, self.routes(url, params))


class SourceTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for p in (mock.patch.object(config, "MIN_REQUEST_INTERVAL", 0), mock.patch.object(fetch.time, "sleep")):
            p.start()
            self.addCleanup(p.stop)

    def client(self, session, **kw):
        return fetch.ArcticShiftClient(cache_dir=self.tmp.name, session=session, **kw)


def wiki_routes(url, params):
    if url.endswith("/wikis/list"):
        return fixture("reddit_wiki_list_bodyweightfitness.json")
    if "paths" in params:
        wanted = set(params["paths"].split(","))
        return {"data": [p for p in fixture("reddit_wiki_paths_bodyweightfitness.json")["data"] if p["path"] in wanted]}
    return fixture("reddit_wiki_bodyweightfitness.json")


class WikiTest(SourceTestCase):
    def test_paths(self):
        self.assertEqual(wiki.wiki_path("/r/bodyweightfitness/wiki/exercises/handstand"),
                         ("bodyweightfitness", "/wiki/exercises/handstand"))
        self.assertTrue(wiki.is_content_path("/r/bodyweightfitness/wiki/exercises/handstand"))
        self.assertFalse(wiki.is_content_path("/r/bodyweightfitness/wiki/edit/exercises/pullup"))
        self.assertFalse(wiki.is_content_path("/r/bodyweightfitness/wiki/revisions/kb/recommended_routine"))
        self.assertFalse(wiki.is_content_path("/r/bodyweightfitness/wiki/automoderator-schedule"))
        self.assertIsNone(wiki.wiki_path("/user/x/wiki"))

    def test_page_evidence(self):
        page = next(p for p in fixture("reddit_wiki_bodyweightfitness.json")["data"]
                    if p["path"].endswith("/exercises/handstand"))
        ev = wiki.page_evidence(page, "bodyweightfitness")
        self.assertEqual(len(ev), 9)  # 9 lines of the page link YouTube
        first = ev[0]
        self.assertEqual(first["kind"], "wiki")
        self.assertEqual(first["author"], "wiki:bodyweightfitness")
        self.assertEqual(first["path"], "/wiki/exercises/handstand")
        self.assertEqual(first["url"], "https://www.reddit.com/r/bodyweightfitness/wiki/exercises/handstand")
        wall = next(e for e in ev if "H2KSIMwNOfA" in e["body"])
        self.assertIn("Progression", wall["body"])  # nearest heading is kept as context

    def test_collect_paginates_beyond_first_100_and_skips_plumbing(self):
        session = RecordedSession(wiki_routes)
        ev = wiki.collect(self.client(session), ["bodyweightfitness"])
        paths = {e["path"] for e in ev}
        self.assertIn("/wiki/exercises/handstand", paths)
        self.assertIn("/wiki/exercises/planche", paths)  # only reachable via the /list + paths= batches
        self.assertFalse(any("automoderator" in p for p in paths))
        batches = [p["paths"].split(",") for _, p in session.calls if "paths" in p]
        self.assertTrue(batches and all(len(b) <= config.WIKI_PATHS_PER_REQUEST for b in batches))
        self.assertFalse(any("/wiki/edit/" in p for b in batches for p in b))
        # Cached: a rerun makes no requests.
        n = len(session.calls)
        self.assertEqual(len(wiki.collect(self.client(session), ["bodyweightfitness"])), len(ev))
        self.assertEqual(len(session.calls), n)

    def test_failing_subreddit_is_skipped(self):
        session = RecordedSession(wiki_routes, status=422)
        with mock.patch.object(config, "MAX_RETRIES", 2):
            self.assertEqual(wiki.collect(self.client(session), ["bodyweightfitness", "Fitness"]), [])

    def test_wiki_aggregation_boost_and_tags(self):
        page = next(p for p in fixture("reddit_wiki_bodyweightfitness.json")["data"]
                    if p["path"].endswith("/exercises/handstand"))
        polish = next(p for p in fixture("reddit_wiki_bodyweightfitness.json")["data"]
                      if p["path"].endswith("/exercises/handstand_polish"))
        ev = wiki.page_evidence(polish, "bodyweightfitness") + wiki.page_evidence(page, "bodyweightfitness")
        vid = "H2KSIMwNOfA"
        videos = build.aggregate(ev)
        v = videos[vid]
        self.assertEqual(v.recommenders, 1)  # one wiki, however many pages/lines
        self.assertEqual(v.wikis, {"bodyweightfitness": "/wiki/exercises/handstand"})  # canonical page wins
        self.assertTrue(build.passes_threshold(v))
        self.assertEqual(v.score, build.rank_score(1, 0) + config.WIKI_BOOST)
        self.assertGreater(v.first_seen, 0)

        class FakeOEmbed(build.OEmbed):
            def __init__(self):
                pass

            def lookup(self, ids):
                return {i: {"title": "Wall Handstand", "channel": "BWF"} for i in ids}

        data = build.build(ev, ["bodyweightfitness"], oembed=FakeOEmbed())
        rec = next(r for r in data["videos"] if r["id"] == vid)
        self.assertEqual(rec["sources"], [{"kind": "wiki", "subreddit": "bodyweightfitness",
                                           "path": "/wiki/exercises/handstand",
                                           "url": "https://www.reddit.com/r/bodyweightfitness/wiki/exercises/handstand"}])
        self.assertEqual(rec["comments"], [])  # wiki lines are not shown as comments
        self.assertIn("skill", rec["tags"]["goal"])
        self.assertIn("upper", rec["tags"]["body"])


class PostsTest(SourceTestCase):
    def posts(self):
        return fixture("reddit_posts_flexibility.json")["data"]

    def by_title(self, start):
        return next(p for p in self.posts() if p["title"].startswith(start))

    def test_classification(self):
        link = next(p for p in self.posts() if p["title"] == "YouTube")  # link post, percent-encoded URL
        ev = posts.post_evidence(link)
        self.assertEqual(ev["kind"], "post")
        self.assertEqual(ev["link_id"], f"t3_{link['id']}")
        self.assertEqual([r.id for r in build.extract_videos(ev["body"])], ["BmAtgabQx0M"])

        ask = self.by_title("Good YouTube tutorials for at home stretching?")
        self.assertIsNone(posts.post_evidence(ask))  # a question is not a recommendation
        self.assertTrue(posts.is_request(ask))
        self.assertTrue(posts.is_request(self.by_title("Best YouTube channels for flexibility")))
        self.assertTrue(posts.is_request(self.by_title("Sharing flexibility YouTubers!")))
        self.assertFalse(posts.is_request(self.by_title("Video recommendations")))  # no replies

    def test_self_promotion_is_not_evidence(self):
        post = {"id": "x", "title": "Check out my new mobility video", "url": "https://youtu.be/AAAAAAAAAAA",
                "selftext": "", "score": 50, "num_comments": 0, "created_utc": 1, "subreddit": "flexibility"}
        self.assertTrue(posts.is_self_promo(post))
        self.assertIsNone(posts.post_evidence(post))

    def test_collect_and_permalink(self):
        def routes(url, params):
            self.assertTrue(url.endswith("/posts/search"))
            self.assertEqual(params["query"], "youtube")
            self.assertNotIn("url", params)
            rows = [p for p in self.posts() if params["after"] <= p["created_utc"] < params["before"]]
            return {"data": rows[: params["limit"]]}

        session = RecordedSession(routes)
        ev, threads = posts.collect(self.client(session), ["flexibility"], max_posts=300)
        self.assertEqual(sorted(e["id"] for e in ev), sorted(f"t3_{p['id']}" for p in self.posts()
                                                             if posts.post_evidence(p)))
        self.assertTrue(threads)
        counts = [next(p["num_comments"] for p in self.posts() if f"t3_{p['id']}" == link) for link, _ in threads]
        self.assertEqual(counts, sorted(counts, reverse=True))  # most-commented first
        link = next(e for e in ev if "BmAtgabQx0M" in e["body"])
        self.assertEqual(build.permalink(link), f"https://www.reddit.com/r/flexibility/comments/{link['id'][3:]}/")

    def test_post_aggregation(self):
        A = "AAAAAAAAAAA"
        post = posts.post_evidence({"id": "p1", "author": "op", "title": "Great hip routine",
                                    "url": f"https://youtu.be/{A}", "selftext": "", "score": 40,
                                    "num_comments": 3, "created_utc": 1_500_000_000, "subreddit": "yoga"})
        reply = {"id": "c1", "author": "bob", "body": f"+1 https://youtu.be/{A}", "score": 5,
                 "created_utc": 1_500_000_100, "subreddit": "yoga", "link_id": "t3_p1"}
        alone = build.aggregate([post])[A]
        self.assertFalse(build.passes_threshold(alone))  # a lone post (even with karma) is not enough
        v = build.aggregate([post, reply])[A]
        self.assertEqual(v.recommenders, 2)
        self.assertEqual(v.weighted_score, 45)
        self.assertTrue(build.passes_threshold(v))
        self.assertEqual(build.source_entries(v), [
            {"kind": "post", "subreddit": "yoga", "url": "https://www.reddit.com/r/yoga/comments/p1/", "score": 40},
            {"kind": "comment", "subreddit": "yoga", "count": 1},
        ])


class DeadlineTest(SourceTestCase):
    def test_past_deadline_uses_cache_only(self):
        session = RecordedSession(wiki_routes)
        client = self.client(session, deadline=fetch.time.monotonic() - 1)
        self.assertEqual(wiki.collect(client, ["bodyweightfitness"]), [])
        self.assertEqual(session.calls, [])

    def test_backoff_past_deadline_gives_up(self):
        session = RecordedSession(wiki_routes, status=429)
        client = self.client(session, deadline=fetch.time.monotonic() + 5)  # backoff starts at 30 s
        with self.assertRaises(fetch.FetchError):
            client.get_json(config.ARCTIC_SHIFT_WIKIS_URL, {"subreddit": "x"}, window_end=0)
        self.assertEqual(len(session.calls), 1)


if __name__ == "__main__":
    unittest.main()
