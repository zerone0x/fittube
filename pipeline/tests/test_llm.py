"""LLM pass (pipeline/llm.py) against a fake Anthropic client: no network, no key."""

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import build, carryover, config, llm

A, B, C, D, E = "AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC", "DDDDDDDDDDD", "EEEEEEEEEEE"
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
OPTS = {"classify_model": HAIKU, "model": SONNET}
WIKI = {"kind": "wiki", "subreddit": "bodyweightfitness", "path": "/wiki/x",
        "url": "https://www.reddit.com/r/bodyweightfitness/wiki/x"}


def comment(cid, text, score=5, sub="Fitness"):
    return {"text": text, "score": score, "permalink": f"https://www.reddit.com/r/{sub}/comments/p{cid}/_/{cid}/"}


def video(vid, title, rec, comments=(), sources=None, karma=10, **kw):
    sources = sources if sources is not None else [{"kind": "comment", "subreddit": "Fitness", "count": len(comments)}]
    v = {"id": vid, "title": title, "channel": "ch", "thumbnail": "", "recommenders": rec,
         "score": round(build.rank_score(rec, karma) + build.sources_boost(sources), 1), "first_seen": "2015-01-01",
         "subreddits": ["Fitness"], "comments": list(comments), "timestamps": [],
         "tags": {"goal": ["strength"], "body": [], "equipment": [], "level": "any", "type": "tutorial"},
         "sources": sources}
    v.update(kw)
    return v


def dataset():
    return {"generated_at": "x", "sources": ["Fitness"], "videos": [
        # Good tutorial, two recommenders, one of them an anti-recommendation.
        video(A, "How to squat", 4, [comment("a1", "This video fixed my squat", 40),
                                     comment("a2", "avoid this guy, terrible advice", 3)]),
        # Single recommender whose only comment cites the video negatively.
        video(B, "Robert Oberst on deadlifts", 1, [comment("b1", "avoid: Oberst says you shouldn't deadlift", 12)]),
        # Music video linked in a training log.
        video(C, "Electric Callboy - PUMP IT", 3, [comment("c1", "my warmup song", 8), comment("c2", "banger", 5)]),
        # Wiki-listed: never dropped, even if stage B rejects it.
        video(D, "PUMP IT wiki edition", 2, [], sources=[WIKI], karma=0),
        # Stage B returns garbage: keyword tags stay.
        video(E, "Hamstring stretch BROKEN", 3, [comment("e1", "great stretch", 9)]),
    ]}


def responder(params):
    """Plays the model: stage A rejects texts with 'avoid'/'song'/'banger'; stage B rejects 'PUMP IT'."""
    user = params["messages"][0]["content"]
    tool = "classify_evidence" if params["system"].startswith("You review evidence") else "review_video"
    if tool == "classify_evidence":
        items = []
        for n, text in re.findall(r"^\[(\d+)\] [^:]+: (.*)$", user, re.M):
            bad = re.search(r"avoid|song|banger", text)
            items.append({"index": int(n), "is_recommendation": not bad,
                          "sentiment": "negative" if "avoid" in text else "positive",
                          "quality_flags": ["off-topic"] if bad else [], "reason": "test"})
        out = {"comments": items}
    elif "BROKEN" in user:
        out = {"keep": True, "goal": ["not-a-goal"]}
    else:
        music = "PUMP IT" in user
        out = {"keep": not music, "reject_reason": "non-fitness" if music else "none",
               "goal": [] if music else ["mobility", "mobility"], "body": ["legs"], "equipment": ["none"],
               "level": "beginner", "type": "tutorial", "summary": "  A short   summary. "}
    if "output_config" in params:  # JSON-schema structured output: the answer is text
        content = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": json.dumps(out)}]
    else:
        content = [{"type": "tool_use", "name": tool, "input": out}]
    return {"model": params["model"], "content": content, "usage": {"input_tokens": 1000, "output_tokens": 100}}


class BadRequest(Exception):
    status_code = 400


class FakeBatches:
    def __init__(self, polls_before_end=1):
        self.polls_before_end = polls_before_end
        self.submitted = {}
        self.polls = {}

    def create(self, requests):
        bid = f"msgbatch_{len(self.submitted) + 1}"
        self.submitted[bid] = requests
        self.polls[bid] = 0
        return {"id": bid}

    def retrieve(self, bid):
        self.polls[bid] += 1
        ended = self.polls[bid] > self.polls_before_end
        return {"processing_status": "ended" if ended else "in_progress",
                "request_counts": {"processing": 0 if ended else len(self.submitted[bid]), "succeeded": 0}}

    def results(self, bid):
        for r in self.submitted[bid]:
            yield {"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": responder(r["params"])}}


class FakeClient:
    def __init__(self, polls_before_end=1):
        self.calls = []
        self.messages = self
        self.batches = FakeBatches(polls_before_end)

    def create(self, **params):
        self.calls.append(params)
        if "sonnet" in params["model"] and params.get("tool_choice", {}).get("type") == "tool":
            raise BadRequest('tool_choice: type "tool" and "any" are not supported for this model.')
        return responder(params)


class LLMTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_path = Path(self.tmp.name) / "llm_cache.json"

    def cache(self):
        return llm.LLMCache(self.cache_path)

    def run_pass(self, data, client=None, clock=None, **opts):
        opts = {"classify_model": HAIKU, "model": SONNET, **opts}
        cache = self.cache()
        runner = None
        if client is not None:
            t = [0.0]

            def sleep(s):
                t[0] += s

            runner = llm.Runner(client, cache, deadline=opts.pop("deadline", None), sleep=sleep,
                                clock=clock or (lambda: t[0]), poll_seconds=10)
        return llm.run_pass(data, cache, client, llm.Options(**opts), runner=runner)


class ValidationTest(unittest.TestCase):
    def test_validate_b(self):
        ok = {"keep": True, "reject_reason": "spam", "goal": ["strength", "strength"], "body": [], "level": "any",
              "equipment": ["none", "barbell"], "type": "tutorial", "summary": "x" * 500}
        v = llm.validate_b(ok)
        self.assertEqual(v["reject_reason"], "none")  # keep wins
        self.assertEqual(v["tags"]["goal"], ["strength"])
        self.assertEqual(v["tags"]["equipment"], ["barbell"])  # "none" contradicts a barbell
        self.assertLessEqual(len(v["summary"]), llm.SUMMARY_CHARS)
        self.assertEqual(llm.validate_b({**ok, "keep": False, "reject_reason": "none"})["reject_reason"], "non-fitness")
        for bad in (None, [], {**ok, "keep": "yes"}, {**ok, "level": "pro"}, {**ok, "body": ["toes"]},
                    {k: x for k, x in ok.items() if k != "goal"}):
            self.assertIsNone(llm.validate_b(bad))

    def test_summary_debris_is_stripped(self):
        ok = {"keep": True, "reject_reason": "none", "goal": [], "body": [], "equipment": [], "level": "any",
              "type": "tutorial"}
        cases = {
            "Shows how to row with a flat back.}": "Shows how to row with a flat back.",
            "Exercises to do at a desk or while riding in a car.\u201d}": "Exercises to do at a desk or while riding in a car.",
            "Easing bicep-related shoulder pain.}</br>": "Easing bicep-related shoulder pain.",
            "Going back to previous weights.}</parameter>Wait.</br>": "Going back to previous weights.",
            "Strength training basics.\u201d}<br>```json</br>": "Strength training basics.",
            "Explains why calves lag as an IFBB pro.</br>": "Explains why calves lag as an IFBB pro.",
            'A "dead bug" core drill for beginners.': 'A "dead bug" core drill for beginners.',
            # Provenance filler goes; clauses that add information stay.
            "A tutorial on how to use the Smith machine, recommended on r/Fitness.":
                "A tutorial on how to use the Smith machine.",
            "Demonstrates the wall scissor, a handstand drill, listed in the r/bodyweightfitness wiki's handstand page.":
                "Demonstrates the wall scissor, a handstand drill.",
            "A short routine for loosening the shoulders, recommended in flexibility and kettlebell communities.":
                "A short routine for loosening the shoulders.",
            "Explains conventional deadlift technique, recommended by commenters as a good resource for learning the lift.":
                "Explains conventional deadlift technique.",
            # Without a clause boundary it may be the point of the sentence: kept unless provenance only.
            "Shows form guidance recommended in strength communities for fixing technique issues.":
                "Shows form guidance recommended in strength communities for fixing technique issues.",
        }
        for raw, want in cases.items():
            self.assertEqual(llm.validate_b({**ok, "summary": raw})["summary"], want, raw)
            self.assertEqual(llm.clean_summary(want), want)  # idempotent
        self.assertEqual(llm.validate_b({**ok, "summary": "}</parameter>"})["summary"], "")
        self.assertEqual(llm.clean_summary(None), "")

    def test_validate_a(self):
        out = {"comments": [
            {"index": 1, "is_recommendation": True, "sentiment": "positive", "quality_flags": ["joke", "nope"],
             "reason": "r"},
            {"index": 7, "is_recommendation": True, "sentiment": "positive", "quality_flags": [], "reason": ""},
            {"index": 2, "is_recommendation": "no", "sentiment": "positive", "quality_flags": [], "reason": ""},
        ]}
        self.assertEqual(llm.validate_a(out, 2), {1: {"is_recommendation": True, "sentiment": "positive",
                                                      "quality_flags": ["joke"], "reason": "r"}})
        self.assertIsNone(llm.validate_a({"comments": []}, 2))
        self.assertIsNone(llm.validate_a("garbage", 2))

    def test_comment_key(self):
        self.assertEqual(llm.comment_key({"permalink": "https://www.reddit.com/r/x/comments/abc/_/def/"}), "t1_def")
        self.assertEqual(llm.comment_key({"permalink": "https://www.reddit.com/r/x/comments/abc/"}), "t3_abc")
        self.assertEqual(llm.comment_key({"permalink": "https://news.ycombinator.com/item?id=42"}), "hn_42")
        self.assertEqual(llm.comment_key({"permalink": "https://t-nation.com/t/some-topic/123/4"}), "tn_123_4")
        self.assertTrue(llm.comment_key({"text": "hi"}).startswith("x_"))

    def test_schemas_use_readme_vocabularies(self):
        props = llm.TOOL_B["input_schema"]["properties"]
        self.assertEqual(props["goal"]["items"]["enum"], ["strength", "hypertrophy", "fat-loss", "mobility", "rehab",
                                                          "endurance", "skill"])
        self.assertEqual(props["level"]["enum"], ["beginner", "intermediate", "advanced", "any"])
        params = llm.request_params("B", "m", "u")
        self.assertEqual(params["tool_choice"], {"type": "tool", "name": "review_video"})

    def test_no_key_means_no_client(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(llm.make_client())


class PassTest(LLMTestCase):
    def test_sync_pass(self):
        client = FakeClient()
        data, report = self.run_pass(dataset(), client)
        ids = [v["id"] for v in data["videos"]]
        self.assertEqual(sorted(ids), sorted([A, D, E]))
        self.assertEqual({d["id"]: d["reason"] for d in report.dropped}, {B: "no-recommendation", C: "non-fitness"})
        a = next(v for v in data["videos"] if v["id"] == A)
        # The anti-recommendation is hidden and no longer counts.
        self.assertEqual([c["text"] for c in a["comments"]], ["This video fixed my squat"])
        self.assertEqual(a["recommenders"], 3)
        self.assertLess(a["score"], build.rank_score(4, 10))
        self.assertEqual(a["llm_rejected"]["recommenders"], 1)
        self.assertEqual(a["tags"], {"goal": ["mobility"], "body": ["legs"], "equipment": ["none"],
                                     "level": "beginner", "type": "tutorial"})
        self.assertEqual(a["summary"], "A short summary.")
        d = next(v for v in data["videos"] if v["id"] == D)
        self.assertEqual(d["recommenders"], 2)  # wiki-listed: kept despite the stage-B rejection
        e = next(v for v in data["videos"] if v["id"] == E)
        self.assertEqual(e["tags"]["goal"], ["strength"])  # invalid output -> keyword tags
        self.assertNotIn("summary", e)
        self.assertEqual(report.invalid_b, 1)
        # Stage A: one request per video with comments; stage B skips B (dropped by A).
        stages = [p["system"][:10] for p in client.calls]
        self.assertEqual(stages.count(llm.SYSTEM_A[:10]), 4)
        # Stage B: 4 videos + 2 rejected style probes (the fake Sonnet refuses forced tool use).
        self.assertEqual(stages.count(llm.SYSTEM_B[:10]), 6)
        self.assertEqual(report.usage["A"].requests, 4)
        self.assertGreater(report.usage["B"].cost, 0)
        cache = json.loads(self.cache_path.read_text())
        self.assertIn(f"{llm.PROMPT_A_VERSION}:{A}:t1_a2", cache["comments"])
        self.assertEqual(cache["videos"][f"{llm.PROMPT_B_VERSION}:{E}"]["invalid"], True)
        self.assertEqual(len(cache["runs"]), 1)

        # Rerun without a key on the same input: identical output from the cache alone.
        again, report2 = self.run_pass(dataset())
        self.assertEqual(again["videos"], data["videos"])
        # And rerunning on its own output is idempotent (strip_llm undoes the adjustment first).
        third, _ = self.run_pass(copy.deepcopy(data))
        self.assertEqual(third["videos"], data["videos"])

    def test_strip_llm_restores_raw_counts(self):
        raw = dataset()
        data, _ = self.run_pass(copy.deepcopy(raw), FakeClient())
        stripped = llm.strip_llm(copy.deepcopy(data))
        a_raw = next(v for v in raw["videos"] if v["id"] == A)
        a = next(v for v in stripped["videos"] if v["id"] == A)
        self.assertEqual((a["recommenders"], a["score"]), (a_raw["recommenders"], a_raw["score"]))
        self.assertEqual(len(a["comments"]), 2)
        self.assertNotIn("llm_rejected", a)

    def test_carry_over_then_pass_is_stable(self):
        data, _ = self.run_pass(dataset(), FakeClient())
        previous = llm.strip_llm(copy.deepcopy(data))
        fresh = {"sources": [], "videos": []}  # a rate-limited run that found nothing new

        class OEmbed:
            cache = {}

            def lookup(self, ids):
                return {i: {"title": "t", "channel": "c"} for i in ids}

        merged = carryover.merge_previous(fresh, previous, oembed=OEmbed())
        again, _ = self.run_pass(merged)
        by_id = {v["id"]: v for v in again["videos"]}
        for v in data["videos"]:
            self.assertEqual((by_id[v["id"]]["recommenders"], by_id[v["id"]]["score"]), (v["recommenders"], v["score"]))

    def test_contested_video_is_dropped(self):
        # Review of #8: "Robert Oberst says you shouldn't do deadlifts", linked once
        # as (bro-science) advice and once to rebut it.
        cache = self.cache()
        v = video(A, "Oberst says you shouldn't deadlift", 2, [comment("o1", "skip deadlifts, see"),
                                                               comment("o2", "I'd push back on this")])
        cache.set_comment(A, "t1_o1", {"is_recommendation": True, "sentiment": "neutral",
                                       "quality_flags": ["bro-science"], "reason": ""})
        cache.set_comment(A, "t1_o2", {"is_recommendation": False, "sentiment": "negative",
                                       "quality_flags": [], "reason": ""})
        self.assertEqual(llm.stage_a_drop(v, cache), "anti-recommendation")
        cache.set_comment(A, "t1_o1", {"is_recommendation": True, "sentiment": "neutral",
                                       "quality_flags": [], "reason": ""})
        self.assertIsNone(llm.stage_a_drop(v, cache))
        self.assertIsNone(llm.stage_a_drop({**v, "sources": [WIKI]}, cache))

    def test_cached_summary_debris_is_cleaned_at_apply_time(self):
        cache = self.cache()
        cache.set_video(A, {"keep": True, "reject_reason": "none", "summary": "Squat basics.}</parameter>Wait.",
                            "tags": {"goal": [], "body": [], "equipment": [], "level": "any", "type": "tutorial"}})
        data = llm.apply_verdicts({"videos": [video(A, "How to squat", 2)]}, cache)
        self.assertEqual(data["videos"][0]["summary"], "Squat basics.")

    def test_rejected_excerpt_removes_its_source(self):
        tn = "https://t-nation.com/t/contest-guess-my-body-fat/292457/1"
        cache = self.cache()
        v = video(A, "What body fat looks like", 3,
                  [comment("g1", "great visual reference"),
                   {"text": "Guess my body fat", "score": 7, "permalink": tn, "site": "t-nation.com"}],
                  sources=[{"kind": "comment", "subreddit": "Fitness"},
                           {"kind": "forum", "site": "t-nation.com", "url": tn, "score": 7, "needs_llm_check": True}])
        raw = copy.deepcopy(v)
        cache.set_comment(A, "t1_g1", {"is_recommendation": True, "sentiment": "neutral", "quality_flags": []})
        cache.set_comment(A, "tn_292457_1", {"is_recommendation": False, "sentiment": "neutral",
                                             "quality_flags": ["personal-clip"]})
        data = llm.apply_verdicts({"videos": [v]}, cache)
        out = data["videos"][0]
        self.assertEqual(out["sources"], [{"kind": "comment", "subreddit": "Fitness"}])
        self.assertEqual([c["text"] for c in out["comments"]], ["great visual reference"])
        restored = llm.strip_llm(copy.deepcopy(data))["videos"][0]
        self.assertEqual(sorted(map(json.dumps, restored["sources"])), sorted(map(json.dumps, raw["sources"])))
        again = llm.apply_verdicts(llm.strip_llm(copy.deepcopy(data)), cache)["videos"][0]
        self.assertEqual(again, out)

    def test_misattributed_excerpts_do_not_drop_a_tutorial(self):
        # Review of #10: "The Cossack Squat" filed under a thread's off-topic replies
        # about high-school track; stage A rejects both, stage B keeps it on content.
        track = [comment("t1", "Sprinting as in track and field in HS", 2, "overcominggravity"),
                 comment("t2", "We used chairs for hurdles", 2, "overcominggravity")]
        tutorial = video(A, "The Cossack Squat", 2, track)
        clip = video(B, "My squat PR", 1, [comment("p1", "new PR today", 3)])
        song = video(C, "Electric Callboy - PUMP IT", 1, [comment("s1", "my warmup song", 3)])
        offtopic = {"is_recommendation": False, "sentiment": "neutral", "quality_flags": ["off-topic"]}
        cache = self.cache()
        for key in ("t1_t1", "t1_t2"):
            cache.set_comment(A, key, offtopic)
        cache.set_comment(B, "t1_p1", {**offtopic, "quality_flags": ["personal-clip"]})
        cache.set_comment(C, "t1_s1", offtopic)
        self.assertTrue(llm.rescuable(tutorial, cache))
        self.assertFalse(llm.rescuable(clip, cache))  # own lift: never reconsidered
        self.assertTrue(llm.rescuable(song, cache))  # reviewed by stage B, which rejects it
        cache.save()

        client = FakeClient()
        data, report = self.run_pass({"sources": [], "videos": [tutorial, clip, song]}, client)
        # Stage B ran only for the rescuable video, without the rejected excerpts.
        b_users = [p["messages"][0]["content"] for p in client.calls if p["system"].startswith("You curate")]
        self.assertTrue(b_users and all("No comment excerpts" in u for u in b_users))
        self.assertFalse([u for u in b_users if "squat PR" in u])
        self.assertEqual([v["id"] for v in data["videos"]], [A])
        out = data["videos"][0]
        self.assertEqual(out["llm_note"], "kept-by-content")
        self.assertEqual((out["recommenders"], out["comments"]), (1, []))
        self.assertEqual([d["id"] for d in report.rescued], [A])
        self.assertEqual(sorted(d["id"] for d in report.dropped), [B, C])
        # Idempotent, and strip_llm restores the raw record.
        again, _ = self.run_pass(copy.deepcopy(data))
        self.assertEqual(again["videos"], data["videos"])
        self.assertNotIn("llm_note", llm.strip_llm(copy.deepcopy(data))["videos"][0])

        # A kept video needs a training title/channel (or a channel known from other videos).
        verdict = {"keep": True, "tags": {"type": "tutorial"}}
        self.assertFalse(llm.kept_by_content(video(A, "Traveling walks", 2, track), cache, verdict))
        self.assertTrue(llm.kept_by_content(video(A, "Traveling walks", 2, track), cache, verdict, {"ch"}))
        # Stage B rejecting it, or calling it motivation, still drops it.
        for verdict in ({"keep": False, "reject_reason": "non-fitness"}, {"keep": True, "reject_reason": "none"}):
            cache = self.cache()
            cache.set_video(A, {**verdict, "summary": "", "tags": {"goal": [], "body": [], "equipment": [],
                                                                  "level": "any", "type": "motivation"}})
            data = llm.apply_verdicts({"videos": [video(A, "The Cossack Squat", 2, track)]}, cache)
            self.assertEqual(data["videos"], [])

    def test_needs_llm_check_cleared_when_accepted(self):
        hn = "https://news.ycombinator.com/item?id=5"
        v = video(A, "Zone 2 explained", 2, [{"text": "great zone 2 explainer", "score": None, "permalink": hn,
                                              "site": "news.ycombinator.com"}],
                  sources=[{"kind": "hn", "site": "news.ycombinator.com", "url": hn, "score": 0, "needs_llm_check": True}],
                  subreddits=[], needs_llm_check=True)
        data, _ = self.run_pass({"sources": [], "videos": [v]}, FakeClient())
        out = data["videos"][0]
        self.assertFalse(out["needs_llm_check"])
        self.assertFalse(out["sources"][0]["needs_llm_check"])

    def test_batch_pass_and_resume(self):
        client = FakeClient(polls_before_end=3)
        data, report = self.run_pass(dataset(), client, force_sync=False)
        # 4 requests <= LLM_SYNC_MAX would go sync; force the batch path via sync_max.
        self.assertTrue(client.calls)

        client = FakeClient(polls_before_end=3)
        self.cache_path.unlink()
        cache = self.cache()
        t = [0.0]
        runner = llm.Runner(client, cache, deadline=15, sync_max=0, poll_seconds=10,
                            sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
        data, report = llm.run_pass(dataset(), cache, client, llm.Options(**OPTS), runner=runner)
        # The deadline passed while stage A was running: it stays pending, nothing dropped.
        # Each stage probes the output style with one direct request: A's first
        # video, then (A answered by its probe) A again for stage B; D, the
        # video without excerpts, goes into a stage-B batch that is pending too.
        self.assertEqual(len(data["videos"]), 5)
        saved = json.loads(self.cache_path.read_text())
        self.assertEqual([(p["stage"], list(p["meta"])) for p in saved["pending"]],
                         [("A", [f"a-{v}" for v in (B, C, E)]), ("B", [f"b-{D}"])])
        self.assertEqual(len(client.calls), 1 + 3)  # B's probe: tool and tool-nothink rejected, json accepted

        # The next run (no deadline) collects that batch instead of resubmitting, then runs stage B.
        cache = self.cache()
        runner = llm.Runner(client, cache, deadline=None, sync_max=0, poll_seconds=10, sleep=lambda s: None)
        data, report = llm.run_pass(dataset(), cache, client, llm.Options(**OPTS), runner=runner)
        self.assertEqual(len(client.batches.submitted), 3)  # stage A, stage B for D, stage B for the rest
        self.assertEqual(sorted(v["id"] for v in data["videos"]), sorted([A, D, E]))
        self.assertEqual(json.loads(self.cache_path.read_text())["pending"], [])
        self.assertEqual(report.usage["A"].requests, 3)

    def test_output_style_fallback(self):
        # The stage-B model rejects forced tool use (400): the probe falls back to
        # JSON-schema output, remembers it and uses it for the batch.
        client = FakeClient()
        cache = self.cache()
        runner = llm.Runner(client, cache, sync_max=0, poll_seconds=0, sleep=lambda s: None)
        data, report = llm.run_pass(dataset(), cache, client, llm.Options(**OPTS), runner=runner)
        self.assertEqual(cache.data["styles"], {"claude-haiku-5-5": "tool", "claude-sonnet-5-5": "json"})
        batch_b = client.batches.submitted["msgbatch_2"]
        self.assertTrue(all("output_config" in r["params"] and "tools" not in r["params"] for r in batch_b))
        self.assertEqual(sorted(v["id"] for v in data["videos"]), sorted([A, D, E]))
        self.assertEqual(next(v for v in data["videos"] if v["id"] == A)["summary"], "A short summary.")
        # Next run starts with the remembered style: no rejected attempts.
        client2 = FakeClient()
        cache.data["videos"].clear()
        runner = llm.Runner(client2, cache, sync_max=0, poll_seconds=0, sleep=lambda s: None)
        llm.run_pass(dataset(), cache, client2, llm.Options(**OPTS), runner=runner)
        self.assertTrue(all("output_config" in p for p in client2.calls))

    def test_tool_input_parsing(self):
        self.assertEqual(llm.tool_input({"content": [{"type": "text", "text": '```json\n{"a": 1}\n```'}]}, "x"), {"a": 1})
        self.assertIsNone(llm.tool_input({"content": [{"type": "text", "text": "sorry"}]}, "x"))
        self.assertIsNone(llm.tool_input({"content": [{"type": "text", "text": "{broken"}]}, "x"))

    def test_budget_guard(self):
        client = FakeClient()
        data, report = self.run_pass(dataset(), client, max_cost=0.0001)
        self.assertEqual(client.calls, [])
        self.assertEqual(len(data["videos"]), 5)
        self.assertEqual(len(report.skipped), 2)
        data, report = self.run_pass(dataset(), client, max_items=1)
        self.assertEqual(len(client.calls), 2 + 2)  # one request per stage (+2 rejected style probes)

    def test_api_errors_fall_back(self):
        class Failing(FakeClient):
            def create(self, **params):
                raise RuntimeError("overloaded")

        data, report = self.run_pass(dataset(), Failing())
        self.assertEqual(len(data["videos"]), 5)
        self.assertEqual(len(report.skipped), 2)  # the probe failed: both stages skipped
        self.assertEqual(data["videos"][0]["tags"]["goal"], ["strength"])

        class Flaky(FakeClient):  # works for the probe, then fails per request
            def create(self, **params):
                if self.calls:
                    raise RuntimeError("overloaded")
                return super().create(**params)

        data, report = self.run_pass(dataset(), Flaky())
        self.assertEqual(len(data["videos"]), 5)
        self.assertEqual(report.usage["A"].errors, 3)


class RunTest(LLMTestCase):
    def test_llm_only_without_key(self):
        from pipeline import run
        src = Path(self.tmp.name) / "videos.json"
        out = Path(self.tmp.name) / "out.json"
        data = dataset()
        data["videos"][0]["comments"].append(
            {"text": "Title https://www.reddit.com/r/Fitness/comments/zz/title/ body", "score": 1,
             "permalink": "https://www.reddit.com/r/Fitness/comments/zz/"})
        src.write_text(json.dumps(data))
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}), \
                mock.patch.object(config, "LLM_CACHE_PATH", self.cache_path), \
                mock.patch.object(llm.LLMCache.__init__, "__defaults__", (self.cache_path,)):
            self.assertEqual(run.main(["--llm-only", "--previous", str(src), "--output", str(out)]), 0)
        written = json.loads(out.read_text())
        self.assertEqual(len(written["videos"]), 5)
        self.assertIn("Title — body", [c["text"] for v in written["videos"] for c in v["comments"]])


    def test_llm_only_starts_from_the_base_snapshot(self):
        from pipeline import run
        base = Path(self.tmp.name) / "base.json"
        out = Path(self.tmp.name) / "out.json"
        base.write_text(json.dumps(dataset()))
        out.write_text(json.dumps({"sources": [], "videos": dataset()["videos"][:1]}))  # an earlier, filtered output
        env = {"ANTHROPIC_API_KEY": ""}
        with mock.patch.dict("os.environ", env), mock.patch.object(llm.LLMCache.__init__, "__defaults__",
                                                                   (self.cache_path,)):
            self.assertEqual(run.main(["--llm-only", "--base", str(base), "--output", str(out)]), 0)
            self.assertEqual(len(json.loads(out.read_text())["videos"]), 5)
            self.assertEqual(json.loads(base.read_text()), dataset())  # the snapshot is never rewritten here
            # Without a snapshot it falls back to --output.
            base.unlink()
            self.assertEqual(run.main(["--llm-only", "--base", str(base), "--output", str(out)]), 0)
            self.assertEqual(len(json.loads(out.read_text())["videos"]), 5)


if __name__ == "__main__":
    unittest.main()
