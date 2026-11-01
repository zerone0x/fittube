import unittest

from pipeline import tagger
from pipeline.tagger import tag_video


class TaggerTest(unittest.TestCase):
    def assertVocab(self, tags):
        self.assertTrue(set(tags["goal"]) <= set(tagger.GOALS))
        self.assertTrue(set(tags["body"]) <= set(tagger.BODY))
        self.assertTrue(set(tags["equipment"]) <= set(tagger.EQUIPMENT))
        self.assertIn(tags["level"], tagger.LEVELS)
        self.assertIn(tags["type"], tagger.TYPES)

    def test_flexibility_follow_along(self):
        tags = tag_video("15 Minute Beginner Hip Flexibility Routine (Follow Along)",
                         ["great for the pancake and middle split"], ["flexibility"])
        self.assertVocab(tags)
        self.assertIn("mobility", tags["goal"])
        self.assertIn("hips", tags["body"])
        self.assertEqual(tags["level"], "beginner")
        self.assertEqual(tags["type"], "follow-along")

    def test_skill_tutorial(self):
        tags = tag_video("How to do a Muscle Up - Step by Step Tutorial", ["practice on a pull-up bar first"],
                         ["bodyweightfitness"])
        self.assertVocab(tags)
        self.assertIn("skill", tags["goal"])
        self.assertIn("bar", tags["equipment"])
        self.assertNotIn("none", tags["equipment"])
        self.assertEqual(tags["type"], "tutorial")

    def test_barbell_strength(self):
        tags = tag_video("Deadlift technique: how to deadlift with proper form", [], ["weightroom"])
        self.assertVocab(tags)
        self.assertIn("strength", tags["goal"])
        self.assertIn("barbell", tags["equipment"])
        self.assertIn("lower", tags["body"])

    def test_rehab_and_neck(self):
        tags = tag_video("Fix Forward Head Posture and Neck Pain", [], ["physicaltherapy"])
        self.assertIn("rehab", tags["goal"])
        self.assertIn("neck", tags["body"])

    def test_explainer(self):
        tags = tag_video("The Science of Muscle Hypertrophy explained", ["great research summary"], ["StrongerByScience"])
        self.assertIn("hypertrophy", tags["goal"])
        self.assertEqual(tags["type"], "explainer")

    def test_kettlebell_prior(self):
        tags = tag_video("Swing practice", ["love this kettlebell swing cue"], ["kettlebell"])
        self.assertIn("kettlebell", tags["equipment"])

    def test_comment_mentions_need_threshold(self):
        # A single comment mention is below threshold; the title wins.
        tags = tag_video("Untitled video", ["my back hurts"], [])
        self.assertEqual(tags, {"goal": [], "body": [], "equipment": [], "level": "any", "type": "tutorial"})

    def test_full_body_inferred(self):
        tags = tag_video("Push-ups, squats and glute bridges for chest, legs and glutes", [], [])
        self.assertEqual(tags["body"][0], "full")

    def test_empty_inputs(self):
        tags = tag_video("", [], [])
        self.assertVocab(tags)
        self.assertEqual(tags["level"], "any")


if __name__ == "__main__":
    unittest.main()
