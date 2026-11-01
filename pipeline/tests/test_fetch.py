import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

from pipeline import config, fetch


class FakeResponse:
    def __init__(self, status, data=None, headers=None):
        self.status_code = status
        self._data = data
        self.headers = headers or {}
        self.text = "" if status == 200 else '{"data":null,"error":"Timeout. Maybe slow down a bit"}'

    def json(self):
        return {"data": self._data, "error": None}


class FakeSession:
    """Serves comments from an in-memory list, honouring after/before/limit."""

    def __init__(self, comments, fail_limits=()):
        self.comments = sorted(comments, key=lambda c: -c["created_utc"])
        self.fail_limits = set(fail_limits)
        self.calls = []
        self.headers = {}

    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        if params["limit"] in self.fail_limits:
            return FakeResponse(422, headers={"x-ratelimit-reset": "0"})
        rows = [c for c in self.comments if params["after"] <= c["created_utc"] < params["before"]]
        return FakeResponse(200, rows[: params["limit"]])


def ts(year, i=0):
    return int(datetime(year, 6, 1, tzinfo=timezone.utc).timestamp()) - i


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patches = [mock.patch.object(config, "MIN_REQUEST_INTERVAL", 0), mock.patch.object(fetch.time, "sleep")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, session):
        return fetch.ArcticShiftClient(cache_dir=self.tmp.name, session=session)

    def test_time_windows(self):
        now = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
        windows = fetch.time_windows(2010, 4, now=now)
        years = [(datetime.fromtimestamp(a, timezone.utc).year, datetime.fromtimestamp(b, timezone.utc).year)
                 for a, b in windows]
        self.assertEqual(years, [(2026, 2030), (2022, 2026), (2018, 2022), (2014, 2018), (2010, 2014)])

    def test_paginates_and_spreads_budget_across_windows(self):
        comments = [{"id": f"a{i}", "created_utc": ts(2025, i)} for i in range(250)]
        comments += [{"id": f"b{i}", "created_utc": ts(2012, i)} for i in range(30)]
        session = FakeSession(comments)
        with mock.patch.object(fetch, "time_windows", return_value=fetch.time_windows(2010, 4, now=ts(2026))):
            got = list(fetch.fetch_subreddit(self.client(session), "x", 300))
        ids = [c["id"] for c in got]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(any(i.startswith("b") for i in ids), "old window must be sampled")
        self.assertLessEqual(len(ids), 300 + config.PAGE_SIZE)

    def test_cache_makes_rerun_free(self):
        comments = [{"id": f"a{i}", "created_utc": ts(2012, i)} for i in range(150)]
        session = FakeSession(comments)
        with mock.patch.object(fetch, "time_windows", return_value=[(ts(2010), ts(2014))]):
            first = list(fetch.fetch_subreddit(self.client(session), "x", 500))
            n_calls = len(session.calls)
            second = list(fetch.fetch_subreddit(self.client(session), "x", 500))
        self.assertEqual(len(first), 150)
        self.assertEqual(first, second)
        self.assertEqual(len(session.calls), n_calls)

    def test_slow_timeout_halves_page_size(self):
        comments = [{"id": f"a{i}", "created_utc": ts(2012, i)} for i in range(120)]
        session = FakeSession(comments, fail_limits={100})
        client = self.client(session)
        clock = iter(range(0, 10_000, 10))  # every request "takes" 10 s -> slow timeout
        with mock.patch.object(fetch.time, "monotonic", side_effect=lambda: next(clock)), \
                mock.patch.object(fetch, "time_windows", return_value=[(ts(2010), ts(2014))]):
            got = list(fetch.fetch_subreddit(client, "x", 120))
        self.assertEqual([c["limit"] for c in session.calls], [100, 50, 50, 50])
        self.assertEqual(len(got), 120)

    def test_offline_never_hits_network(self):
        session = FakeSession([{"id": "a", "created_utc": ts(2012)}])
        client = fetch.ArcticShiftClient(cache_dir=self.tmp.name, session=session, offline=True)
        self.assertEqual(list(fetch.fetch_subreddit(client, "x", 100)), [])
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
