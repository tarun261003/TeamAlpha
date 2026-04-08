# truth_seeker_env/tests/test_web_search_tool.py
"""
Tests for the WebSearchTool and the dual-mode WebSearchProxy.

Tests are organized into three groups:
  1. WebSearchTool unit tests   — snippet parsing, expansion, formatting
  2. WebSearchProxy fallback    — keyword-overlap (Phase 1 parity)
  3. WebSearchProxy real mode   — mocked Serper API calls
"""

import pytest
from unittest.mock import patch, MagicMock

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────────────
# 1. WebSearchTool unit tests
# ──────────────────────────────────────────────────────────────────────────────

from server.web_search_tool import WebSearchTool, WebContent


class TestSnippetParsing:
    """Test _parse_search_snippet static method."""

    def test_splits_on_ellipsis(self):
        snippet = "Python is great ... for data science and machine learning ... short"
        segments = WebSearchTool._parse_search_snippet(snippet)
        # "short" has only 1 word, should be filtered out (need >5 words)
        assert len(segments) >= 1
        assert "for data science and machine learning" in segments[0]

    def test_empty_snippet(self):
        assert WebSearchTool._parse_search_snippet("") == []

    def test_no_ellipsis(self):
        # Entire snippet as one segment, but must have >5 words
        snippet = "This is a short test of the snippet parsing function"
        segments = WebSearchTool._parse_search_snippet(snippet)
        assert len(segments) == 1

    def test_short_segments_filtered(self):
        snippet = "ok ... yes ... this segment has more than five words definitely"
        segments = WebSearchTool._parse_search_snippet(snippet)
        # "ok" and "yes" are too short
        assert len(segments) == 1


class TestSnippetExpansion:
    """Test _expand_search_snippet static method."""

    def test_expands_to_paragraph(self):
        web_content = (
            "First paragraph about cats and their behavior patterns.\n"
            "Second paragraph about Python is great for building modern web applications and APIs.\n"
            "Third paragraph about dogs and their training routines."
        )
        # Each segment after splitting on '...' must have >5 words to pass the filter
        snippet = "Python is great for building modern web applications ... and APIs for production use cases"
        expanded = WebSearchTool._expand_search_snippet(snippet, web_content)
        assert "Python is great for building modern web applications" in expanded

    def test_no_match_returns_empty(self):
        web_content = "Nothing relevant here."
        snippet = "completely unrelated query ... about quantum physics"
        expanded = WebSearchTool._expand_search_snippet(snippet, web_content)
        assert expanded == ""


class TestFormatWebContents:
    """Test _format_web_contents static method."""

    def test_formats_correctly(self):
        contents = [
            WebContent(title="Result 1", content="Content one", url="https://example.com/1"),
            WebContent(title="Result 2", content="Content two", url="https://example.com/2"),
        ]
        formatted = WebSearchTool._format_web_contents(contents, "test query")
        assert "Search results for: test query" in formatted
        assert "[1] Result 1" in formatted
        assert "[2] Result 2" in formatted
        assert "https://example.com/1" in formatted
        assert "Content one" in formatted

    def test_truncates_long_content(self):
        long_content = "x" * 600
        contents = [WebContent(title="Long", content=long_content, url="")]
        formatted = WebSearchTool._format_web_contents(contents, "q")
        assert "..." in formatted  # Should truncate at 500 chars

    def test_empty_results(self):
        formatted = WebSearchTool._format_web_contents([], "q")
        assert "Search results for: q" in formatted


class TestWebSearchToolSearch:
    """Test the main search() method with mocked API."""

    @patch("server.web_search_tool.requests.post")
    def test_search_returns_formatted_results(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Python Docs",
                    "snippet": "Python is an interpreted high-level programming language for general purpose",
                    "link": "https://python.org",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = WebSearchTool(api_key="test-key", snippet_only=True)
        result = tool.search("what is python")
        assert "Python Docs" in result
        assert "python.org" in result

    def test_empty_query_returns_error(self):
        tool = WebSearchTool(api_key="test-key")
        result = tool.search("")
        assert "empty query" in result.lower()

    @patch("server.web_search_tool.requests.post")
    def test_api_failure_returns_error(self, mock_post):
        mock_post.side_effect = Exception("Connection timeout")
        tool = WebSearchTool(api_key="test-key")
        result = tool.search("test query")
        assert "error" in result.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 2. WebSearchProxy — fallback mode tests (Phase 1 parity)
# ──────────────────────────────────────────────────────────────────────────────

from server.web_search import WebSearchProxy

# Unset SERPER_API_KEY to force fallback mode in tests
@pytest.fixture
def fallback_proxy():
    """Create a WebSearchProxy in fallback mode (no API key)."""
    with patch.dict(os.environ, {"SERPER_API_KEY": ""}, clear=False):
        proxy = WebSearchProxy(serper_api_key="")
    return proxy


@pytest.fixture
def loaded_proxy(fallback_proxy):
    """Fallback proxy with sample episode data loaded."""
    fallback_proxy.load_episode_data({
        "keywords": ["pinecone", "serverless", "regions", "us-east-1"],
        "results": [
            {
                "url": "https://docs.pinecone.io/regions",
                "title": "Pinecone Serverless Regions",
                "snippet": "Supported regions include us-east-1 on AWS.",
                "relevance_keywords": ["pinecone", "serverless", "us-east-1", "aws", "regions"],
            },
            {
                "url": "https://example.com/other",
                "title": "Unrelated Page",
                "snippet": "This page is about cooking pasta.",
                "relevance_keywords": ["cooking", "pasta", "recipe"],
            },
        ],
    })
    return fallback_proxy


class TestFallbackSearch:
    """Tests for keyword-overlap fallback mode."""

    def test_relevant_query_returns_results(self, loaded_proxy):
        result = loaded_proxy.search("pinecone serverless us-east-1 regions")
        assert "Pinecone Serverless Regions" in result
        assert "us-east-1" in result

    def test_irrelevant_query_returns_no_results(self, loaded_proxy):
        result = loaded_proxy.search("quantum computing algorithms")
        assert "No relevant web results" in result

    def test_empty_query_returns_error(self, loaded_proxy):
        result = loaded_proxy.search("")
        assert "empty query" in result.lower() or "No relevant" in result

    def test_no_data_loaded(self, fallback_proxy):
        result = fallback_proxy.search("anything")
        assert "No web results available" in result

    def test_partial_match_returns_if_above_threshold(self, loaded_proxy):
        # "pinecone regions" should match with enough overlap
        result = loaded_proxy.search("pinecone regions")
        assert "Pinecone" in result


class TestQueryQualityScore:
    """Tests for get_query_quality_score (deterministic, both modes)."""

    def test_perfect_match(self, loaded_proxy):
        score = loaded_proxy.get_query_quality_score(
            "pinecone serverless regions us-east-1"
        )
        assert score == 1.0

    def test_partial_match(self, loaded_proxy):
        score = loaded_proxy.get_query_quality_score("pinecone regions")
        assert 0.0 < score < 1.0

    def test_no_match(self, loaded_proxy):
        score = loaded_proxy.get_query_quality_score("quantum physics")
        assert score == 0.0

    def test_empty_query(self, loaded_proxy):
        score = loaded_proxy.get_query_quality_score("")
        assert score == 0.0

    def test_no_keywords_loaded(self, fallback_proxy):
        score = fallback_proxy.get_query_quality_score("anything")
        assert score == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. WebSearchProxy — real mode tests (mocked Serper API)
# ──────────────────────────────────────────────────────────────────────────────

class TestRealSearchMode:
    """Tests for real search mode with mocked Serper API."""

    @patch.dict(os.environ, {"SERPER_API_KEY": ""}, clear=False)
    @patch("server.web_search_tool.requests.post")
    def test_real_mode_returns_live_results(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Live Result",
                    "snippet": "This is a real search result from the live web search api",
                    "link": "https://live.example.com",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        proxy = WebSearchProxy(serper_api_key="test-real-key")
        proxy.load_episode_data({"keywords": ["test"], "results": []})
        result = proxy.search("test query")
        assert "Live Result" in result

    @patch.dict(os.environ, {"SERPER_API_KEY": ""}, clear=False)
    @patch("server.web_search_tool.requests.post")
    def test_real_mode_falls_back_on_api_error(self, mock_post):
        mock_post.side_effect = Exception("API down")

        proxy = WebSearchProxy(serper_api_key="test-real-key")
        proxy.load_episode_data({
            "keywords": ["python"],
            "results": [
                {
                    "url": "https://fallback.example.com",
                    "title": "Fallback Result",
                    "snippet": "This came from fallback mode",
                    "relevance_keywords": ["python", "programming", "language"],
                },
            ],
        })
        result = proxy.search("python programming")
        assert "Fallback Result" in result

    @patch.dict(os.environ, {"SERPER_API_KEY": ""}, clear=False)
    def test_real_mode_query_quality_still_deterministic(self):
        """get_query_quality_score always uses topic_keywords, even in real mode."""
        proxy = WebSearchProxy(serper_api_key="test-real-key")
        proxy.load_episode_data({
            "keywords": ["alpha", "beta", "gamma", "delta"],
            "results": [],
        })
        score = proxy.get_query_quality_score("alpha beta")
        assert score == 0.5  # 2/4 keywords matched
