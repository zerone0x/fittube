import unittest

from pipeline import build, carryover, config

A, B, C, D = "AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC", "DDDDDDDDDDD"


def record(id, rec=2, score=None, comments=(), subreddits=("Fitness",), sources=None, **kw):
    r = {
        "id": id, "title": f"title {id}", "channel": "ch", "thumbnail": "", "recommenders": rec,
        "score": score if score is not None else build.rank_score(rec, 10), "first_seen": "2015-01-01",
        "subreddits": list(subreddits), "comments": list(comments), "timestamps": [],
        "tags": {"goal": ["strength"], "body": [], "equipment": [], "level": "any", "type": "tutorial"},
    }
    if sources is not None:
        r["sources"] = sources
    r.update(kw)
    return r


class FakeOEmbed:
    def __init__(self, gone=()):
        self.cache = {i: None for i in gone}

    def lookup(self, ids):
        return {i: (None if i in self.cache else {"title": "t", "channel": "c"}) for i in ids}


BOT = ("Videos in this thread: Watch Playlist ▶ VIDEO|COMMENT -|- (1) StrongFirst SFG Kettlebell Level I "
       "Technique Standards|+2 -")


class CarryOverTest(unittest.TestCase):
    def test_bot_body_filter(self):
        self.assertTrue(build.is_bot_body(BOT))
        self.assertTrue(build.is_bot_body("**Videos in this thread:** [Watch Playlist ▶](http://subtletv.com/x)"))
        self.assertFalse(build.is_bot_body("I posted the videos in this thread earlier, watch the first one"))
        videos = build.aggregate([{"id": "1", "author": "human_looking_name", "score": 5, "subreddit": "Fitness",
                                   "created_utc": 1, "link_id": "t3_x", "body": f"{BOT} https://youtu.be/{A}"}])
        self.assertEqual(videos, {})

    def test_previous_videos_are_carried_over(self):
        previous = {"sources": ["Fitness"], "videos": [record(A, 5), record(B, 3)]}
        fresh = {"sources": ["yoga"], "videos": [record(C, 2, subreddits=["yoga"],
                                                       sources=[{"kind": "comment", "subreddit": "yoga", "count": 2}])]}
        out = carryover.merge_previous(fresh, previous, oembed=FakeOEmbed())
        self.assertEqual([v["id"] for v in out["videos"]], [A, B, C])
        self.assertEqual(out["sources"], ["yoga", "Fitness"])
        # Legacy records gain a sources list derived from their subreddits.
        self.assertEqual(out["videos"][0]["sources"], [{"kind": "comment", "subreddit": "Fitness"}])

    def test_videos_gone_from_youtube_are_dropped(self):
        previous = {"videos": [record(A), record(B)]}
        out = carryover.merge_previous({"sources": [], "videos": []}, previous, oembed=FakeOEmbed(gone=[B]))
        self.assertEqual([v["id"] for v in out["videos"]], [A])

    def test_bot_excerpts_are_stripped(self):
        human = {"text": "great cue", "score": 4, "permalink": "https://www.reddit.com/r/x/comments/1/_/a/"}
        bot = {"text": BOT, "score": 9, "permalink": "https://www.reddit.com/r/x/comments/1/_/b/"}
        previous = {"videos": [record(A, 3, comments=[bot, human]), record(B, 1, comments=[bot])]}
        out = carryover.merge_previous({"sources": [], "videos": []}, previous, oembed=FakeOEmbed())
        self.assertEqual([v["id"] for v in out["videos"]], [A])  # B was only linked by the bot
        v = out["videos"][0]
        self.assertEqual(v["comments"], [human])
        self.assertEqual(v["recommenders"], 2)
        self.assertEqual(v["score"], round(build.rank_score(3, 10) - 10, 1))

    def test_moderation_wiki_listings_are_dropped(self):
        mod = {"kind": "wiki", "subreddit": "Fitness", "path": "/wiki/moderation",
               "url": "https://www.reddit.com/r/Fitness/wiki/moderation"}
        faq = {"kind": "wiki", "subreddit": "weightroom", "path": "/wiki/faq",
               "url": "https://www.reddit.com/r/weightroom/wiki/faq"}
        meme = record(A, 1, score=build.rank_score(1, 0) + build.wiki_boost(1), sources=[mod])
        both = record(B, 3, score=build.rank_score(3, 10) + build.wiki_boost(2),
                      sources=[faq, mod, {"kind": "comment", "subreddit": "Fitness", "count": 1}])
        out = carryover.merge_previous({"sources": [], "videos": []}, {"videos": [meme, both]}, oembed=FakeOEmbed())
        self.assertEqual([v["id"] for v in out["videos"]], [B])  # A was only listed on the moderation page
        v = out["videos"][0]
        self.assertEqual([s.get("path") for s in v["sources"]], ["/wiki/faq", None])
        self.assertEqual(v["recommenders"], 2)
        self.assertEqual(v["score"], round(build.rank_score(2, 10) + build.wiki_boost(1), 1))

    def test_merge_overlapping_record(self):
        prev = record(A, 5, score=70.0, comments=[{"text": "old", "score": 50, "permalink": "https://www.reddit.com/1"}],
                      subreddits=["Fitness"], first_seen="2012-03-04",
                      timestamps=[{"t": 30, "label": ""}])
        wiki_src = {"kind": "wiki", "subreddit": "bodyweightfitness", "path": "/wiki/x",
                    "url": "https://www.reddit.com/r/bodyweightfitness/wiki/x"}
        fresh = record(A, 2, score=build.rank_score(2, 3) + config.WIKI_BOOST,
                       comments=[{"text": "new", "score": 3, "permalink": "https://www.reddit.com/2"}],
                       subreddits=["bodyweightfitness", "yoga"], first_seen="2020-01-01",
                       timestamps=[{"t": 30, "label": "x"}, {"t": 90, "label": ""}],
                       sources=[wiki_src, {"kind": "comment", "subreddit": "yoga", "count": 1}])
        out = carryover.merge_previous({"sources": [], "videos": [fresh]}, {"videos": [prev]}, oembed=FakeOEmbed())
        v = out["videos"][0]
        self.assertEqual(v["recommenders"], 5 + 1)  # community max + the wiki
        # The wiki is one more recommender (10 points) on top of WIKI_BOOST (review of #9).
        self.assertEqual(v["score"], 70.0 + 10 + config.WIKI_BOOST)
        self.assertEqual(v["first_seen"], "2012-03-04")
        self.assertEqual(v["subreddits"], ["bodyweightfitness", "yoga", "Fitness"])
        self.assertEqual([c["text"] for c in v["comments"]], ["old", "new"])
        self.assertEqual([t["t"] for t in v["timestamps"]], [30, 90])
        self.assertEqual([s["kind"] for s in v["sources"]], ["wiki", "comment", "comment"])
        # Merging again with itself is stable (idempotent refresh).
        again = carryover.merge_previous({"sources": [], "videos": [fresh]}, out, oembed=FakeOEmbed())
        self.assertEqual(again["videos"][0]["recommenders"], 6)
        self.assertEqual(again["videos"][0]["score"], v["score"])

    def test_gaining_a_wiki_listing_scores_its_recommender(self):
        # Regression (review of #9, e.g. 2z8JmcrW-As): the previous record has
        # more community evidence, the fresh one adds a wiki listing. The
        # merged score must equal the formula for the merged recommenders.
        wiki_src = {"kind": "wiki", "subreddit": "bodyweightfitness", "path": "/wiki/x",
                    "url": "https://www.reddit.com/r/bodyweightfitness/wiki/x"}
        prev = record(A, 4, score=build.rank_score(4, 5))
        fresh = record(A, 3, score=build.rank_score(3, 2) + config.WIKI_BOOST, sources=[wiki_src])
        v = carryover.merge_records(carryover.clean_previous(prev), fresh)
        self.assertEqual(v["recommenders"], 5)
        self.assertEqual(v["score"], round(build.rank_score(5, 5) + config.WIKI_BOOST, 1))

    def test_union_sources(self):
        merged = carryover.union_sources(
            [{"kind": "comment", "subreddit": "Fitness", "count": 2},
             {"kind": "hn", "site": "news.ycombinator.com", "url": "https://news.ycombinator.com/item?id=1"}],
            [{"kind": "comment", "subreddit": "fitness", "count": 5},
             {"kind": "post", "subreddit": "yoga", "url": "u", "score": 3},
             {"kind": "post", "subreddit": "yoga", "url": "u", "score": 9}],
        )
        self.assertEqual(merged, [
            {"kind": "post", "subreddit": "yoga", "url": "u", "score": 9},
            {"kind": "comment", "subreddit": "Fitness", "count": 5},
            {"kind": "hn", "site": "news.ycombinator.com", "url": "https://news.ycombinator.com/item?id=1"},
        ])

    def test_wiki_boost_survives_external_merge(self):
        from pipeline.sources import Mention
        from pipeline.sources.merge import merge_external, strip_external
        wiki_src = {"kind": "wiki", "subreddit": "bodyweightfitness", "path": "/wiki/x",
                    "url": "https://www.reddit.com/r/bodyweightfitness/wiki/x"}
        base = build.rank_score(1, 0) + config.WIKI_BOOST
        data = {"sources": [], "videos": [record(A, 1, score=base, subreddits=["bodyweightfitness"],
                                                 sources=[wiki_src])]}
        mentions = [Mention(video_id=A, kind="hn", site="news.ycombinator.com", author=f"hn:{u}", text="good",
                            url=f"https://news.ycombinator.com/item?id={i}", created_utc=1_600_000_000)
                    for i, u in enumerate(["pg", "dang"])]
        merged = merge_external(data, mentions, oembed=FakeOEmbed())
        v = merged["videos"][0]
        self.assertEqual(v["recommenders"], 3)
        self.assertEqual(v["source_stats"]["reddit"]["sum_score"], 0)  # the boost is not karma
        self.assertEqual(v["score"], build.rank_score(3, 0) + config.WIKI_BOOST)
        stripped = strip_external(merged)["videos"][0]
        self.assertEqual((stripped["recommenders"], stripped["score"]), (1, base))
        self.assertEqual(stripped["sources"], [wiki_src])

    def test_repair_excerpts_restores_underscores(self):
        vid = "_QnwAoesJvQ"
        texts = ["Source: http://www.youtube.com/watch?v=QnwAoesJvQt=475 Good form",
                 "Watch: http://youtu.be/QnwAoesJvQ and https://youtu.be/QnwAoesJvQt=12",
                 "Unrelated https://youtu.be/abcdefghijk and watch?v=ZZZZZZZZZZt=5"]
        videos = [record(vid, comments=[{"text": t, "score": 1, "permalink": f"p{i}"} for i, t in enumerate(texts)])]
        self.assertEqual(carryover.repair_excerpts(videos), 2)
        self.assertEqual([c["text"] for c in videos[0]["comments"]], [
            "Source: http://www.youtube.com/watch?v=_QnwAoesJvQ&t=475 Good form",
            "Watch: http://youtu.be/_QnwAoesJvQ and https://youtu.be/_QnwAoesJvQ?t=12",
            texts[2]])
        self.assertEqual(carryover.repair_excerpts(videos), 0)  # idempotent

    def test_repair_excerpts_restores_lost_timestamp_separator_for_any_id(self):
        # #11: only ids with "_" were repaired; "v=bt4m4zLioRwt=268" stayed broken.
        texts = ["work on your setup http://www.youtube.com/watch?v=bt4m4zLioRwt=268",
                 "see https://youtu.be/bt4m4zLioRwt=12 and https://www.youtube.com/watch?v=bt4m4zLioRw&t=5",
                 "a 10-character fragment: watch?v=ZZZZZZZZZZt=5"]
        videos = [record("bt4m4zLioRw", comments=[{"text": t, "score": 1, "permalink": f"p{i}"}
                                                  for i, t in enumerate(texts)])]
        self.assertEqual(carryover.repair_excerpts(videos), 2)
        self.assertEqual([c["text"] for c in videos[0]["comments"]], [
            "work on your setup http://www.youtube.com/watch?v=bt4m4zLioRw&t=268",
            "see https://youtu.be/bt4m4zLioRw?t=12 and https://www.youtube.com/watch?v=bt4m4zLioRw&t=5",
            texts[2]])
        self.assertEqual(carryover.repair_excerpts(videos), 0)  # idempotent

    def test_no_previous(self):
        fresh = {"sources": [], "videos": [record(A)]}
        self.assertIs(carryover.merge_previous(fresh, None), fresh)


if __name__ == "__main__":
    unittest.main()
