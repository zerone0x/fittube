import unittest

from pipeline.extract import extract_videos, parse_duration, parse_youtube_url, text_timestamps

VID = "dQw4w9WgXcQ"


class ParseUrlTest(unittest.TestCase):
    def test_url_forms(self):
        cases = [
            f"https://www.youtube.com/watch?v={VID}",
            f"http://youtube.com/watch?feature=share&v={VID}",
            f"https://m.youtube.com/watch?v={VID}&list=PL123",
            f"https://music.youtube.com/watch?v={VID}",
            f"www.youtube.com/watch?v={VID}",
            f"https://youtu.be/{VID}",
            f"youtu.be/{VID}?si=abcdef",
            f"https://www.youtube.com/shorts/{VID}",
            f"https://www.youtube.com/embed/{VID}?rel=0",
            f"https://www.youtube-nocookie.com/embed/{VID}",
            f"https://www.youtube.com/v/{VID}",
            f"https://www.youtube.com/live/{VID}",
        ]
        for url in cases:
            with self.subTest(url=url):
                self.assertEqual(parse_youtube_url(url), (VID, None))

    def test_non_video_urls(self):
        for url in ["https://www.youtube.com/channel/UC123", "https://www.youtube.com/user/foo",
                    "https://www.youtube.com/playlist?list=PL123", "https://youtu.be/short",
                    "https://www.youtube.com/watch?v=tooshort"]:
            with self.subTest(url=url):
                self.assertIsNone(parse_youtube_url(url))

    def test_start_params(self):
        self.assertEqual(parse_youtube_url(f"https://youtu.be/{VID}?t=95"), (VID, 95))
        self.assertEqual(parse_youtube_url(f"https://www.youtube.com/watch?v={VID}&t=1m35s"), (VID, 95))
        self.assertEqual(parse_youtube_url(f"https://www.youtube.com/watch?v={VID}#t=1h2m3s"), (VID, 3723))
        self.assertEqual(parse_youtube_url(f"https://www.youtube.com/embed/{VID}?start=42"), (VID, 42))
        self.assertEqual(parse_youtube_url(f"https://youtu.be/{VID}?t=95s."), (VID, 95))

    def test_parse_duration(self):
        self.assertEqual(parse_duration("95"), 95)
        self.assertEqual(parse_duration("2m"), 120)
        self.assertEqual(parse_duration("1h"), 3600)
        self.assertIsNone(parse_duration("abc"))
        self.assertIsNone(parse_duration(""))


class ExtractVideosTest(unittest.TestCase):
    def test_markdown_link_with_label(self):
        refs = extract_videos(f"Try [the hip hinge part](https://youtu.be/{VID}?t=120).")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].id, VID)
        self.assertEqual(refs[0].timestamps, [(120, "the hip hinge part")])

    def test_markdown_escaped_underscore_and_entities(self):
        vid = "ab_cd-EF_12"
        refs = extract_videos(f"https://www.youtube.com/watch?v=ab\\_cd-EF\\_12&amp;t=30")
        self.assertEqual([(r.id, r.timestamps) for r in refs], [(vid, [(30, "")])])

    def test_multiple_videos_dedup(self):
        body = f"See https://youtu.be/{VID} and https://www.youtube.com/watch?v={VID} and https://youtu.be/AAAAAAAAAAA"
        self.assertEqual([r.id for r in extract_videos(body)], [VID, "AAAAAAAAAAA"])

    def test_text_timestamps_only_for_single_video(self):
        single = extract_videos(f"https://youtu.be/{VID}\n\n2:30 - scap pull-ups\n1:02:03 the cooldown")
        self.assertEqual(single[0].timestamps, [(150, "scap pull-ups"), (3723, "the cooldown")])
        multi = extract_videos(f"https://youtu.be/{VID} https://youtu.be/AAAAAAAAAAA at 2:30")
        self.assertTrue(all(r.timestamps == [] for r in multi))

    def test_no_youtube(self):
        self.assertEqual(extract_videos("just do the recommended routine"), [])
        self.assertEqual(extract_videos("I watch youtube a lot"), [])


class TextTimestampTest(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(text_timestamps("check 4:05 for the cue"), [(245, "for the cue")])
        self.assertEqual(text_timestamps("at 12:00"), [(720, "")])
        self.assertEqual(text_timestamps("the good part: 1:02:03"), [(3723, "the good part")])

    def test_rejects_non_timestamps(self):
        self.assertEqual(text_timestamps("I train at 6:30 am"), [])
        self.assertEqual(text_timestamps("ratio 3:1 and 1:75"), [])
        self.assertEqual(text_timestamps("score 10:2"), [])
        self.assertEqual(text_timestamps("John 3:16:99"), [])

    def test_generic_labels_dropped(self):
        self.assertEqual(text_timestamps("in this video at 2:20"), [(140, "")])

    def test_merge_close_timestamps(self):
        refs = extract_videos(f"https://youtu.be/{VID}?t=126 the couch stretch at 2:07 - couch stretch")
        self.assertEqual(refs[0].timestamps, [(126, "couch stretch")])

    def test_multiple_per_line(self):
        self.assertEqual(text_timestamps("0:45 warmup, 3:10 main set"), [(45, "warmup"), (190, "main set")])


if __name__ == "__main__":
    unittest.main()
