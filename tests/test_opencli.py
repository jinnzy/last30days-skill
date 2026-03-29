"""Tests for opencli wrapper module."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lib import opencli


class TestCommandResolution(unittest.TestCase):
    def test_prefers_explicit_command(self):
        self.assertEqual(
            opencli._resolve_command_prefix("npx -y @jackwener/opencli"),
            ["npx", "-y", "@jackwener/opencli"],
        )

    def test_uses_installed_binary(self):
        with mock.patch.object(opencli.shutil, "which", side_effect=lambda name: "/usr/bin/opencli" if name == "opencli" else None):
            self.assertEqual(opencli._resolve_command_prefix(), ["opencli"])


class TestSearchReddit(unittest.TestCase):
    def test_normalizes_results(self):
        raw_results = [
            {
                "title": "Claude Code tips",
                "subreddit": "r/ClaudeAI",
                "author": "alice",
                "score": 123,
                "comments": 45,
                "url": "https://www.reddit.com/r/ClaudeAI/comments/abc/claude_code_tips/",
            }
        ]

        with mock.patch("lib.reddit.expand_reddit_queries", return_value=["claude code"]), \
             mock.patch.object(opencli, "_run_opencli", return_value=raw_results):
            result = opencli.search_reddit("claude code tips", "2026-03-01", "2026-03-29")

        self.assertEqual(result["source"], "opencli")
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["subreddit"], "r/ClaudeAI")
        self.assertEqual(result["items"][0]["engagement"]["score"], 123)


class TestSearchX(unittest.TestCase):
    def test_filters_old_posts(self):
        raw_results = [
            {
                "id": "1",
                "author": "openai",
                "text": "fresh post",
                "created_at": "Wed Mar 26 12:00:00 +0000 2026",
                "likes": 10,
                "url": "https://x.com/openai/status/1",
            },
            {
                "id": "2",
                "author": "openai",
                "text": "old post",
                "created_at": "Wed Jan 01 12:00:00 +0000 2026",
                "likes": 5,
                "url": "https://x.com/openai/status/2",
            },
        ]

        with mock.patch.object(opencli, "_x_query_variants", return_value=["openai since:2026-03-01"]), \
             mock.patch.object(opencli, "_run_opencli", return_value=raw_results):
            result = opencli.search_x("openai", "2026-03-01", "2026-03-29")

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["author_handle"], "openai")
        self.assertEqual(result["items"][0]["date"], "2026-03-26")

    def test_search_handles_builds_handle_queries(self):
        raw_results = [
            {
                "author": "openai",
                "text": "hello",
                "created_at": "Wed Mar 26 12:00:00 +0000 2026",
                "url": "https://x.com/openai/status/1",
            }
        ]

        with mock.patch.object(opencli, "_run_opencli", return_value=raw_results) as run_mock:
            items = opencli.search_x_handles(["openai"], "gpt-5", "2026-03-01", count_per=2)

        self.assertEqual(len(items), 1)
        called_args = run_mock.call_args.args[0]
        self.assertEqual(called_args[:3], ["twitter", "search", "from:openai gpt-5 since:2026-03-01"])


class TestSearchWeb(unittest.TestCase):
    def test_normalizes_google_results(self):
        raw_results = [
            {
                "type": "result",
                "title": "Launch post",
                "url": "https://example.com/blog/launch",
                "snippet": "A recent launch write-up.",
            },
            {
                "type": "result",
                "title": "Reddit mirror",
                "url": "https://www.reddit.com/r/test/comments/abc/test/",
                "snippet": "Should be excluded.",
            },
        ]

        with mock.patch.object(opencli, "_run_opencli", return_value=raw_results):
            result = opencli.search_web("opencli", "2026-03-01", "2026-03-29")

        self.assertEqual(result["source"], "opencli")
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["source_domain"], "example.com")


if __name__ == "__main__":
    unittest.main()
