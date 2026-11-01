import copy
import unittest

from pipeline.build import rank_score
from pipeline.sources import Mention
from pipeline.sources.merge import merge_external, reddit_sum_score, strip_external

A = "AAAAAAAAAAA"  # already on the site (Reddit)
B = "BBBBBBBBBBB"  # new, two HN recommenders
C = "CCCCCCCCCCC"  # new, single mention -> below threshold
D = "DDDDDDDDDDD"  # new, only linked by its own channel
E = "EEEEEEEEEEE"  # new, two HN recommenders but an off-topic title
F = "FFFFFFFFFFF"  # new, one forum poster with many likes
G = "GGGGGGGGGGG"  # new, T-Nation linking its own channel


class FakeOEmbed:
    def __init__(self, meta):
        self.meta = meta
        self.asked = []

    def lookup(self, ids):
        self.asked += ids
        return {i: self.meta.get(i) for i in ids}


def mention(vid, author, kind="hn", score=None, created=1_600_000_000, negative=False, text="great squat video",
            url=None, ts=()):
    site = "news.ycombinator.com" if kind == "hn" else "t-nation.com"
    return Mention(video_id=vid, kind=kind, site=site, author=f"{'hn' if kind == 'hn' else 'tnation'}:{author}",
                   text=text, url=url or f"https://{site}/{author}/{vid}", created_utc=created, score=score,
                   negative=negative, timestamps=list(ts))


def base_data():
    return {
        "generated_at": "2026-11-01T00:00:00Z",
        "sources": ["bodyweightfitness"],
        "videos": [{
            "id": A, "title": "Squat tutorial", "channel": "Coach", "thumbnail": "x",
            "recommenders": 3, "score": rank_score(3, 40), "first_seen": "2016-01-01",
            "subreddits": ["bodyweightfitness"],
            "comments": [{"text": "reddit says hi", "score": 5, "permalink": "https://www.reddit.com/r/x/comments/1/_/2/"}],
            "timestamps": [{"t": 30, "label": ""}],
            "tags": {"goal": ["strength"], "body": [], "equipment": [], "level": "any", "type": "tutorial"},
        }],
    }


META = {B: {"title": "Deadlift form", "channel": "Lifter"}, C: {"title": "c", "channel": "c"},
        D: {"title": "My channel", "channel": "Self Promo"}, E: {"title": "Don't Talk to the Police", "channel": "Law"},
        F: {"title": "My PR", "channel": "Someone"}, G: {"title": "Push-ups", "channel": "Testosterone Nation"}}


def mentions():
    return [
        mention(A, "pg", text="best squat cues 1:05", ts=[(65, "")], created=1_300_000_000),
        mention(A, "alice", kind="forum", score=4, text="this squat video helped"),
        mention(A, "alice", kind="forum", score=2, url="https://t-nation.com/second"),  # same author twice
        mention(A, "hater", kind="hn", negative=True, text="terrible form"),
        mention(B, "x"), mention(B, "y"),
        mention(C, "z"),
        mention(E, "x", text="exercise your right https://youtu.be/x"), mention(E, "y"),
        mention(F, "fan", kind="forum", score=30), mention(F, "fan", kind="forum", score=12, url="https://t-nation.com/2"),
        mention(G, "t_nation", kind="forum", score=9), mention(G, "reader", kind="forum", score=1),
        mention(D, "selfpromo", kind="forum", score=1), mention(D, "self_promo", kind="hn"),
    ]


class MergeTest(unittest.TestCase):
    def merged(self, data=None, ms=None):
        oembed = FakeOEmbed(META)
        return merge_external(data or base_data(), mentions() if ms is None else ms, oembed=oembed), oembed

    def test_existing_video_gains_namespaced_recommenders(self):
        data, oembed = self.merged()
        a = next(v for v in data["videos"] if v["id"] == A)
        self.assertEqual(a["recommenders"], 3 + 2)  # pg + alice (negative mention does not count)
        self.assertAlmostEqual(a["score"], rank_score(5, 40 + 4 + 2 + 0.25 * 0), delta=0.2)
        self.assertEqual(a["first_seen"], "2011-03-13")
        self.assertEqual(a["subreddits"], ["bodyweightfitness"])
        self.assertEqual(a["tags"]["goal"], ["strength"])  # Reddit tags kept
        self.assertEqual([t["t"] for t in a["timestamps"]], [30, 65])
        sites = [c.get("site") for c in a["comments"]]
        self.assertEqual(sites[0], None)  # Reddit comments first
        self.assertEqual(sorted(s for s in sites if s), ["news.ycombinator.com", "t-nation.com"])
        self.assertEqual({s["kind"] for s in a["sources"]}, {"hn", "forum"})
        self.assertTrue(all(s["needs_llm_check"] for s in a["sources"]))
        self.assertEqual(a["source_stats"]["reddit"]["recommenders"], 3)
        self.assertEqual(a["source_stats"]["forum"]["recommenders"], 1)
        self.assertFalse(a["needs_llm_check"])
        self.assertNotIn(A, oembed.asked)

    def test_new_videos_need_threshold_and_no_self_promotion(self):
        data, _ = self.merged()
        ids = [v["id"] for v in data["videos"]]
        self.assertIn(B, ids)
        self.assertNotIn(C, ids)  # one recommender, no score
        self.assertNotIn(D, ids)  # both mentions by the channel itself
        self.assertNotIn(E, ids)  # HN-only and the title is not about fitness
        self.assertNotIn(F, ids)  # likes alone do not admit a new video
        self.assertNotIn(G, ids)  # the site's own channel
        b = next(v for v in data["videos"] if v["id"] == B)
        self.assertEqual(b["recommenders"], 2)
        self.assertEqual(b["subreddits"], [])
        self.assertTrue(b["needs_llm_check"])
        self.assertEqual(b["first_seen"], "2020-09-13")
        self.assertEqual(len(b["comments"]), 1)  # identical texts are de-duplicated
        self.assertEqual(b["comments"][0]["permalink"].split("/")[2], "news.ycombinator.com")
        self.assertEqual(set(b["tags"]), {"goal", "body", "equipment", "level", "type"})
        self.assertEqual(data["sources"], ["bodyweightfitness", "news.ycombinator.com", "t-nation.com"])
        self.assertEqual([v["score"] for v in data["videos"]], sorted((v["score"] for v in data["videos"]), reverse=True))

    def test_merge_is_idempotent(self):
        once, _ = self.merged()
        twice, _ = self.merged(copy.deepcopy(once))
        self.assertEqual(once, twice)

    def test_strip_restores_reddit_only_data(self):
        data, _ = self.merged()
        stripped = strip_external(copy.deepcopy(data))
        self.assertEqual(stripped, base_data())

    def test_no_mentions_is_noop(self):
        data, oembed = self.merged(ms=[])
        self.assertEqual(data, base_data())
        self.assertEqual(oembed.asked, [])

    def test_reddit_sum_inverse(self):
        for rec, s in [(2, 0), (5, 12.5), (12, 300)]:
            self.assertAlmostEqual(reddit_sum_score(rank_score(rec, s), rec), s, delta=max(0.1, s * 0.03))


if __name__ == "__main__":
    unittest.main()
