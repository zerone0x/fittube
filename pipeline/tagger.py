"""Keyword-rule tagger.

Maps a video's title, the comments recommending it and the subreddits it was
recommended in onto the tag vocabularies from the README data contract.

This module is deliberately self-contained: ``tag_video`` takes plain strings
and returns the ``tags`` dict, so it can later be swapped for an LLM tagger
without touching the rest of the pipeline.

Scoring: each rule match in the title is worth ``TITLE_WEIGHT`` points; each
comment that matches is worth 1 point; a subreddit prior adds ``PRIOR_WEIGHT``.
Tags reaching ``THRESHOLD`` are kept (at most ``MAX_PER_CATEGORY``, best first).

Hints (subreddit wiki paths such as ``/wiki/exercises/handstand``) are matched
against the same rules plus ``PATH_HINTS`` and are worth ``HINT_WEIGHT`` each:
a moderator filing a video under "handstand" is decisive on its own.
"""

from __future__ import annotations

import re
from typing import Iterable

GOALS = ["strength", "hypertrophy", "fat-loss", "mobility", "rehab", "endurance", "skill"]
BODY = ["full", "upper", "lower", "core", "back", "chest", "shoulders", "arms", "legs", "glutes", "neck", "hips"]
EQUIPMENT = ["none", "dumbbell", "barbell", "kettlebell", "band", "rings", "bar", "machine"]
LEVELS = ["beginner", "intermediate", "advanced", "any"]
TYPES = ["tutorial", "follow-along", "explainer", "program", "motivation"]

TITLE_WEIGHT = 3
PRIOR_WEIGHT = 1
HINT_WEIGHT = 2
THRESHOLD = 2
MAX_PER_CATEGORY = {"goal": 3, "body": 4, "equipment": 3}


def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


RULES: dict[str, dict[str, re.Pattern]] = {
    "goal": {
        "strength": _rx(r"strength", r"stronger", r"powerlift\w*", r"1rm", r"one rep max", r"max(?:imal)? strength",
                        r"5x5", r"starting strength", r"stronglifts", r"5/3/1", r"531", r"powerlifting", r"deadlift\w*", r"bench press", r"back squats?",
                        r"front squats?", r"overhead press", r"weighted", r"pull[- ]?ups?", r"chin[- ]?ups?",
                        r"push[- ]?ups?"),
        "hypertrophy": _rx(r"hypertrophy", r"muscle growth", r"build(?:ing)? muscle", r"muscle gain", r"bodybuild\w*",
                           r"gains", r"bulk(?:ing)?", r"aesthetic\w*", r"muscle size", r"training volume",
                           r"sets to failure", r"muscle mass"),
        "fat-loss": _rx(r"fat loss", r"lose fat", r"losing fat", r"weight loss", r"lose weight", r"losing weight", r"cutting", r"on a cut",
                        r"calorie deficit", r"deficit", r"burn fat", r"fat burn\w*", r"body ?fat", r"diet(?:ing)?", r"tdee"),
        "mobility": _rx(r"mobility", r"flexib\w+", r"stretch\w*", r"splits?", r"pancake", r"pike", r"yoga", r"range of motion",
                        r"rom", r"limber", r"tight (?:hips|hamstrings|shoulders)", r"hamstrings? flexibility", r"middle split",
                        r"front split", r"bridge", r"pnf", r"loaded stretch\w*"),
        "rehab": _rx(r"rehab\w*", r"physio\w*", r"physical therap\w*", r"injur\w*", r"pain", r"tendin\w+", r"tendon\w*",
                     r"impingement", r"sciatica", r"posture", r"prehab", r"knee pain", r"back pain",
                     r"shoulder pain", r"elbow pain", r"wrist pain", r"rotator cuff", r"herniat\w+", r"bulging disc"),
        "endurance": _rx(r"endurance", r"cardio", r"running", r"runners?", r"marathon", r"conditioning", r"hiit", r"zone 2",
                         r"aerobic", r"vo2 ?max", r"stamina", r"cycling", r"rowing", r"swim\w*", r"jump rope"),
        "skill": _rx(r"handstands?", r"planche", r"front lever", r"back lever", r"muscle[- ]ups?", r"human flag", r"l-?sit",
                     r"v-?sit", r"calisthenics skills?", r"skills?", r"gymnastic\w*", r"one arm (?:pull[- ]?up|push[- ]?up|chin[- ]?up)",
                     r"pistol squats?", r"press to handstand", r"manna", r"olympic lift\w*", r"snatch", r"clean and jerk",
                     r"juggl\w+", r"parkour", r"acrobat\w*"),
    },
    "body": {
        "full": _rx(r"full[- ]body", r"total body", r"whole body", r"compound (?:lifts?|movements?|exercises?)", r"burpees?"),
        "upper": _rx(r"upper[- ]body", r"upper day", r"push[- ]?ups?", r"pull[- ]?ups?", r"chin[- ]?ups?", r"dips?", r"rows",
                     r"overhead press", r"ohp", r"handstand push[- ]?ups?", r"hspu"),
        "lower": _rx(r"lower[- ]body", r"lower day", r"leg day", r"squats?", r"deadlifts?", r"lunges?", r"rdls?", r"step[- ]ups?"),
        "core": _rx(r"core", r"abs", r"abdominals?", r"planks?", r"hollow (?:body|hold)", r"l-?sit", r"ab wheel", r"crunch\w*",
                    r"obliques?", r"dragon flag", r"leg raises?", r"sit[- ]?ups?", r"six pack"),
        "back": _rx(r"back (?:day|workout|exercises?|training|muscles?|strength|width|thickness)", r"upper back", r"lower back",
                    r"lats?", r"rows", r"pull[- ]?ups?", r"chin[- ]?ups?", r"deadlifts?", r"spine", r"lumbar",
                    r"thoracic", r"traps?", r"rhomboids?", r"front lever", r"back lever"),
        "chest": _rx(r"chest", r"pecs?", r"pectorals?", r"bench(?: press)?", r"push[- ]?ups?", r"dips?", r"flyes?", r"flys?"),
        "shoulders": _rx(r"shoulders?", r"delts?", r"deltoids?", r"rotator cuff", r"overhead press", r"ohp", r"lateral raises?",
                         r"face pulls?", r"scapula\w*", r"scap", r"handstands?", r"planche", r"shoulder dislocates?"),
        "arms": _rx(r"arms?", r"biceps?", r"triceps?", r"forearms?", r"curls?", r"elbows?", r"wrists?", r"grip"),
        "legs": _rx(r"legs?", r"leg day", r"quads?", r"quadriceps", r"hamstrings?", r"calf", r"calves", r"squats?", r"lunges?",
                    r"pistol squats?", r"knees?", r"ankles?", r"tibialis", r"nordic curls?"),
        "glutes": _rx(r"glutes?", r"gluteus", r"butt", r"booty", r"hip thrusts?", r"glute bridges?"),
        "neck": _rx(r"neck", r"cervical", r"forward head", r"chin tucks?"),
        "hips": _rx(r"hips?", r"hip flexors?", r"psoas", r"pancake", r"middle split", r"splits?", r"pelvi\w+", r"adductors?",
                    r"pigeon", r"90/90", r"piriformis"),
    },
    "equipment": {
        "none": _rx(r"bodyweight", r"body weight", r"no equipment", r"calisthenics", r"at home", r"home workout",
                    r"without (?:equipment|weights|a gym)", r"no gym"),
        "dumbbell": _rx(r"dumbbells?", r"db (?:press|row|curl|bench)s?"),
        "barbell": _rx(r"barbells?", r"bench press", r"deadlifts?", r"back squats?", r"front squats?", r"overhead press",
                       r"ohp", r"power cleans?", r"snatch", r"clean and jerk", r"5x5", r"starting strength", r"stronglifts", r"5/3/1"),
        "kettlebell": _rx(r"kettlebells?", r"kb", r"kbs", r"kettlebell swings?", r"turkish get[- ]?ups?", r"tgu"),
        "band": _rx(r"resistance bands?", r"bands?", r"banded", r"theraband", r"band pull[- ]?aparts?"),
        "rings": _rx(r"gymnastic rings", r"rings", r"ring (?:dips?|rows?|muscle[- ]ups?|push[- ]?ups?|support)"),
        "bar": _rx(r"pull[- ]?up bar", r"pull[- ]?ups?", r"chin[- ]?ups?", r"bar hangs?", r"dead hangs?", r"muscle[- ]ups?",
                   r"front lever", r"parallel bars?", r"parallettes?", r"dip bars?"),
        "machine": _rx(r"machines?", r"cable\w*", r"leg press", r"lat pull[- ]?downs?", r"smith machine", r"pec deck",
                       r"leg extensions?", r"hack squats?", r"treadmill", r"rower", r"elliptical"),
    },
    "level": {
        "beginner": _rx(r"beginners?", r"newbies?", r"novices?", r"first time", r"for starters", r"getting started",
                        r"start(?:ing)? out", r"basics", r"101", r"recommended routine", r"couch to", r"never (?:worked out|exercised)"),
        "intermediate": _rx(r"intermediates?", r"next level", r"plateau\w*", r"progress(?:ion)? beyond"),
        "advanced": _rx(r"advanced", r"elite", r"expert", r"one arm", r"planche", r"front lever", r"human flag", r"iron cross",
                        r"maltese", r"competition", r"world record"),
    },
    "type": {
        "follow-along": _rx(r"follow[- ]?along", r"routine", r"workout", r"min(?:ute)? (?:workout|routine|stretch\w*|yoga)",
                            r"\d+ ?min(?:ute)?s?", r"real[- ]time", r"do this every day", r"daily (?:routine|stretch\w*)", r"yoga with"),
        "tutorial": _rx(r"how to", r"tutorial", r"technique", r"form", r"cues?", r"guide", r"step by step", r"learn",
                        r"progressions?", r"mistakes?", r"tips?", r"fix your", r"proper", r"correct(?:ly)?", r"demo\w*"),
        "explainer": _rx(r"science", r"research", r"studies", r"evidence", r"explain\w*", r"why (?:you|your|we|do|does|is|are)", r"myths?",
                         r"the truth", r"what is", r"physiology", r"anatomy", r"lecture", r"podcast", r"interview",
                         r"debunk\w*", r"vs\.?", r"versus", r"optimal", r"mechanism\w*"),
        "program": _rx(r"programs?", r"programming", r"program design", r"(?:training|workout|bro) split", r"plan", r"periodization",
                       r"ppl", r"push pull legs", r"upper ?/ ?lower", r"5x5", r"5/3/1", r"week \d+", r"\d+[- ]week", r"template"),
        "motivation": _rx(r"motivation\w*", r"inspir\w*", r"transformation", r"journey", r"never give up", r"discipline",
                          r"mindset", r"documentary", r"story", r"before and after", r"speech"),
    },
}

# Subreddit priors: recommended in r/X adds PRIOR_WEIGHT to these tags.
SUBREDDIT_PRIORS: dict[str, dict[str, list[str]]] = {
    "flexibility": {"goal": ["mobility"]},
    "physicaltherapy": {"goal": ["rehab"]},
    "kettlebell": {"equipment": ["kettlebell"]},
    "overcominggravity": {"goal": ["skill"], "equipment": ["none"]},
    "bodyweightfitness": {"equipment": ["none"]},
    "weightroom": {"goal": ["strength"]},
    "naturalbodybuilding": {"goal": ["hypertrophy"]},
    "strongerbyscience": {"type": ["explainer"]},
    "yoga": {"goal": ["mobility"]},
    "climbing": {"goal": ["strength"]},
    "climbingtrainer": {"goal": ["strength"]},
    "powerlifting": {"goal": ["strength"], "equipment": ["barbell"]},
    "stronglifts5x5": {"goal": ["strength"], "equipment": ["barbell"]},
    "backpain": {"goal": ["rehab"]},
    "posture": {"goal": ["rehab"]},
    "running": {"goal": ["endurance"]},
    "advancedrunning": {"goal": ["endurance"]},
    "c25k": {"goal": ["endurance"], "level": ["beginner"]},
    "swimming": {"goal": ["endurance"]},
    "bjj": {"goal": ["skill"]},
    "muaythai": {"goal": ["skill"]},
    "kettleballs": {"equipment": ["kettlebell"]},
    "calisthenics": {"equipment": ["none"]},
    "homegym": {},
    "crossfit": {},
    "fitness30plus": {},
    "over40fitness": {},
}

# Wiki path segment -> tags it implies (on top of the keyword RULES).
PATH_HINTS: dict[str, dict[str, list[str]]] = {
    "handstand": {"goal": ["skill"], "body": ["upper", "shoulders"]},
    "planche": {"goal": ["skill"], "body": ["upper", "shoulders"]},
    "lever": {"goal": ["skill"], "body": ["upper", "core"]},
    "muscle-up": {"goal": ["skill"], "body": ["upper"]},
    "l-sit": {"goal": ["skill"], "body": ["core"]},
    "manna": {"goal": ["skill"], "body": ["core"]},
    "stretching": {"goal": ["mobility"]},
    "flexibility": {"goal": ["mobility"]},
    "mobility": {"goal": ["mobility"]},
    "yoga": {"goal": ["mobility"]},
    "wrist": {"goal": ["mobility"], "body": ["arms"]},
    "injur": {"goal": ["rehab"]},
    "pain": {"goal": ["rehab"]},
    "posture": {"goal": ["rehab"]},
    "running": {"goal": ["endurance"]},
    "cardio": {"goal": ["endurance"]},
    "pushup": {"body": ["upper", "chest"]},
    "pullup": {"body": ["upper", "back"], "equipment": ["bar"]},
    "dip": {"body": ["upper", "chest"]},
    "row": {"body": ["upper", "back"]},
    "squat": {"body": ["lower", "legs"]},
    "hinge": {"body": ["lower", "glutes"]},
    "legs": {"body": ["lower", "legs"]},
    "core": {"body": ["core"]},
    "rings": {"equipment": ["rings"]},
    "routine": {"type": ["program"]},
    "program": {"type": ["program"]},
    "beginner": {"level": ["beginner"]},
    "getting_started": {"level": ["beginner"]},
}


# A PATH_HINTS key must start a path word ("core" matches /core-work, not /score).
_HINT_KEY_RE = {k: re.compile(r"(?<![a-z])" + re.escape(k), re.IGNORECASE) for k in PATH_HINTS}


def _hint_text(path: str) -> str:
    """``/wiki/exercises/one-arm-pullup`` -> ``"exercises one arm pullup"``."""
    return re.sub(r"[/_\-]+", " ", path.lower().removeprefix("/wiki")).strip()


def _score(category: str, title: str, comments: Iterable[str], subreddits: Iterable[str],
           hints: Iterable[str] = ()) -> dict[str, float]:
    scores: dict[str, float] = {}
    rules = RULES[category]
    hint_texts = [_hint_text(h) for h in hints]
    for tag, rx in rules.items():
        s = 0.0
        if rx.search(title):
            s += TITLE_WEIGHT
        for text in comments:
            if rx.search(text):
                s += 1
        if any(rx.search(h) for h in hint_texts):
            s += HINT_WEIGHT
        if s:
            scores[tag] = s
    for sub in subreddits:
        for tag in SUBREDDIT_PRIORS.get(sub.lower(), {}).get(category, []):
            scores[tag] = scores.get(tag, 0) + PRIOR_WEIGHT
    hinted = {tag for h in hints for key, cats in PATH_HINTS.items() if _HINT_KEY_RE[key].search(h)
              for tag in cats.get(category, [])}
    for tag in hinted:
        scores[tag] = scores.get(tag, 0) + HINT_WEIGHT
    return scores


def _pick(scores: dict[str, float], order: list[str], limit: int) -> list[str]:
    kept = [t for t, s in scores.items() if s >= THRESHOLD]
    kept.sort(key=lambda t: (-scores[t], order.index(t)))
    return kept[:limit]


def tag_video(title: str, comments: Iterable[str] = (), subreddits: Iterable[str] = (),
              hints: Iterable[str] = ()) -> dict:
    """Return ``{"goal": [...], "body": [...], "equipment": [...], "level": str, "type": str}``.

    ``hints`` are subreddit wiki paths the video is listed under.
    """
    title = title or ""
    comments = [c for c in comments if c]
    subreddits = list(subreddits)
    hints = [h for h in hints if h]

    goal = _pick(_score("goal", title, comments, subreddits, hints), GOALS, MAX_PER_CATEGORY["goal"])

    body_scores = _score("body", title, comments, subreddits, hints)
    body = _pick(body_scores, BODY, MAX_PER_CATEGORY["body"])
    has_upper = "upper" in body or bool({"chest", "shoulders", "arms", "back"} & set(body))
    has_lower = "lower" in body or bool({"legs", "glutes", "hips"} & set(body))
    if "full" not in body and has_upper and has_lower and len(body) >= 3:
        body = ["full"] + body[: MAX_PER_CATEGORY["body"] - 1]

    equipment = _pick(_score("equipment", title, comments, subreddits, hints), EQUIPMENT, MAX_PER_CATEGORY["equipment"])
    if "none" in equipment and len(equipment) > 1:
        # "none" contradicts a specific piece of kit; keep it only if it outranks
        # all of them (bodyweight videos often mention a bar or rings).
        others = [e for e in equipment if e != "none"]
        if equipment[0] != "none":
            equipment = others

    level_scores = _score("level", title, comments, subreddits, hints)
    level_pick = _pick(level_scores, LEVELS, 1)
    level = level_pick[0] if level_pick else "any"

    type_scores = _score("type", title, comments, subreddits, hints)
    type_pick = _pick(type_scores, TYPES, 1)
    if type_pick:
        vtype = type_pick[0]
    elif type_scores:
        vtype = max(type_scores, key=lambda t: (type_scores[t], -TYPES.index(t)))
    else:
        vtype = "tutorial"

    return {"goal": goal, "body": body, "equipment": equipment, "level": level, "type": vtype}
