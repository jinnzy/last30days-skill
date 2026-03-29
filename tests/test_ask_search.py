"""Tests for ask-search wrapper module."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lib import ask_search


class TestCommandResolution(unittest.TestCase):
    def test_prefers_explicit_command(self):
        self.assertEqual(
            ask_search._resolve_command_prefix("/usr/local/bin/ask-search"),
            ["/usr/local/bin/ask-search"],
        )

    def test_uses_installed_binary(self):
        with mock.patch.object(ask_search.shutil, "which", return_value="/usr/bin/ask-search"):
            self.assertEqual(ask_search._resolve_command_prefix(), ["ask-search"])


class TestSearchWeb(unittest.TestCase):
    def test_normalizes_results(self):
        raw = {
            "query": "opencli",
            "results": [
                {
                    "title": "Launch post",
                    "url": "https://example.com/post",
                    "content": "Recent launch write-up.",
                    "engines": ["google", "brave"],
                },
                {
                    "title": "Reddit mirror",
                    "url": "https://www.reddit.com/r/test/comments/abc/test/",
                    "content": "Should be excluded.",
                },
            ],
        }

        with mock.patch.object(ask_search, "_run_ask_search", return_value=raw):
            result = ask_search.search_web("opencli", "2026-03-01", "2026-03-29")

        self.assertEqual(result["source"], "ask-search")
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["source_domain"], "example.com")
        self.assertIn("google,brave", result["items"][0]["why_relevant"])


if __name__ == "__main__":
    unittest.main()
