import tempfile
import unittest
from pathlib import Path

from pipeline import build


def comment(id, author, body, score=5, sub="bodyweightfitness", created=1_500_000_000):
    return {"id": id, "author": author, "body": body, "score": score, "subreddit": sub,
            "created_utc": created, "link_id": "t3_abc"}


A = "AAAAAAAAAAA"
B = "BBBBBBBBBBB"
C = "CCCCCCCCCCC"


class FakeOEmbed(build.OEmbed):
    def __init__(self, meta):
        self.meta = meta

    def lookup(self, ids):
        return {i: self.meta.get(i) for i in ids}


class BuildTest(unittest.TestCase):
    def test_noise_filters(self):
        self.assertTrue(build.is_bot("AutoModerator"))
        self.assertTrue(build.is_bot("[deleted]"))
        self.assertTrue(build.is_bot("RemindMeBot"))
        self.assertFalse(build.is_bot("robotics_fan"))
        self.assertTrue(build.is_negative("Don't watch this guy, terrible form"))
        self.assertTrue(build.is_negative("total bro science"))
        self.assertFalse(build.is_negative("Great form cues in this one"))

    def test_aggregate(self):
        videos = build.aggregate([
            comment("1", "alice", f"https://youtu.be/{A}?t=30", score=10, created=1_600_000_000),
            comment("2", "Alice", f"again https://youtu.be/{A}", score=2),  # same author, different case
            comment("3", "bob", f"https://youtu.be/{A}", score=3, sub="Fitness"),
            comment("4", "carol", f"don't watch https://youtu.be/{A}", score=8),  # negative
            comment("5", "dave", f"https://youtu.be/{A}", score=0),  # below min score
            comment("6", "AutoModerator", f"https://youtu.be/{A}", score=50),  # bot
        ])
        v = videos[A]
        self.assertEqual(v.recommenders, 2)
        self.assertEqual(v.weighted_score, 10 + 2 + 3 + 8 * 0.25)
        self.assertEqual(v.subreddits, {"bodyweightfitness": 3, "Fitness": 1})
        self.assertEqual(v.first_seen, 1_500_000_000)
        self.assertEqual(v.timestamps, [(30, "")])

    def test_blocklisted_video_is_ignored(self):
        videos = build.aggregate([comment("1", "alice", f"lol https://youtu.be/dQw4w9WgXcQ https://youtu.be/{A}")])
        self.assertEqual(list(videos), [A])

    def test_multi_video_comment_splits_score(self):
        videos = build.aggregate([comment("1", "alice", f"https://youtu.be/{A} https://youtu.be/{B}", score=10)])
        self.assertEqual(videos[A].weighted_score, 5)
        self.assertEqual(videos[B].recommenders, 1)

    def test_threshold_and_score(self):
        videos = build.aggregate([
            comment("1", "alice", f"https://youtu.be/{A}", score=1),
            comment("2", "bob", f"https://youtu.be/{A}", score=1),
            comment("3", "carol", f"https://youtu.be/{B}", score=12),
            comment("4", "dave", f"https://youtu.be/{C}", score=3),
        ])
        self.assertTrue(build.passes_threshold(videos[A]))  # 2 recommenders
        self.assertTrue(build.passes_threshold(videos[B]))  # score >= 10
        self.assertFalse(build.passes_threshold(videos[C]))
        self.assertEqual(build.rank_score(2, 2), 25.5)
        self.assertGreater(build.rank_score(3, 0), build.rank_score(1, 20))

    def test_excerpt(self):
        self.assertEqual(build.excerpt("> quoted\n\n**Watch** [this](https://youtu.be/x) now"), "Watch this now")
        long = build.excerpt("word " * 200)
        self.assertLessEqual(len(long), 280)
        self.assertTrue(long.endswith("…"))

    def test_excerpt_keeps_underscores_in_urls(self):
        # Review of #8: youtu.be/K4dmZ5_n6uU was shown as youtu.be/K4dmZ5n6uU.
        self.assertEqual(build.excerpt("_Great_ one: **https://youtu.be/K4dmZ5_n6uU** and snake_case"),
                         "Great one: https://youtu.be/K4dmZ5_n6uU and snake_case")
        self.assertEqual(build.excerpt("see youtu.be/_K4dmZ5n6u_ ok"), "see youtu.be/_K4dmZ5n6u_ ok")

    def test_post_excerpt_drops_the_url_line(self):
        # Review of #9: post excerpts began with "title https://...".
        body = "Great hamstring routine\nhttps://www.youtube.com/watch?v=AAAAAAAAAAA\nDoing this daily."
        self.assertEqual(build.excerpt(build.post_text(body)), "Great hamstring routine \u2014 Doing this daily.")
        self.assertEqual(build.post_text("Title\nhttps://youtu.be/AAAAAAAAAAA\n"), "Title")
        videos = build.aggregate([{"id": "t3_p", "kind": "post", "author": "a", "score": 5, "subreddit": "yoga",
                                   "created_utc": 1, "link_id": "t3_p", "body": body}])
        self.assertEqual(videos["AAAAAAAAAAA"].comments[0]["text"], "Great hamstring routine \u2014 Doing this daily.")

    def test_build_contract(self):
        comments = [
            comment("1", "alice", f"Best tutorial: https://youtu.be/{A}\n\n2:15 - the hollow body cue", score=40),
            comment("2", "bob", f"+1 https://www.youtube.com/watch?v={A}", score=4, sub="Fitness", created=1_400_000_000),
            comment("3", "carol", f"https://youtu.be/{B}", score=30),  # will be "deleted" on YouTube
            comment("4", "dave", f"https://youtu.be/{C}", score=1),  # below threshold
        ]
        oembed = FakeOEmbed({A: {"title": "Hollow Body Hold Tutorial for Beginners", "channel": "Coach"}})
        data = build.build(comments, ["bodyweightfitness", "Fitness"], oembed=oembed)
        self.assertEqual(data["sources"], ["bodyweightfitness", "Fitness"])
        self.assertRegex(data["generated_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual([v["id"] for v in data["videos"]], [A])
        v = data["videos"][0]
        self.assertEqual(set(v), {"id", "title", "channel", "thumbnail", "recommenders", "score", "first_seen",
                                  "subreddits", "comments", "timestamps", "tags", "sources"})
        self.assertEqual(v["sources"], [{"kind": "comment", "subreddit": "bodyweightfitness", "count": 1},
                                        {"kind": "comment", "subreddit": "Fitness", "count": 1}])
        self.assertEqual(v["thumbnail"], f"https://i.ytimg.com/vi/{A}/hqdefault.jpg")
        self.assertEqual(v["recommenders"], 2)
        self.assertEqual(v["first_seen"], "2014-05-13")
        self.assertEqual(v["subreddits"], ["bodyweightfitness", "Fitness"])
        self.assertEqual(v["comments"][0]["score"], 40)
        self.assertTrue(v["comments"][0]["permalink"].startswith("https://www.reddit.com/r/bodyweightfitness/comments/abc/"))
        self.assertEqual(v["timestamps"], [{"t": 135, "label": "the hollow body cue"}])
        self.assertIn("core", v["tags"]["body"])
        self.assertEqual(v["tags"]["level"], "beginner")

    def test_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "site" / "data" / "videos.json"
            build.write({"videos": []}, path)
            self.assertTrue(path.read_text().startswith("{"))


if __name__ == "__main__":
    unittest.main()
