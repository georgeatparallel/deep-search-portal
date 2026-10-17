"""Exercise the optional provider through the real MCP SDK and HTTP requests.

Only the remote server and unrelated search providers are replaced. No API keys
or running services are needed.
"""

import asyncio
import json
import os
import subprocess
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mcp.client import streamable_http

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))
import search_providers as sp


@pytest.mark.parametrize("value,enabled", [
    (None, False), ("false", False), ("0", False), ("true", True), ("1", True), ("YES", True),
])
def test_environment_opt_in(value, enabled):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "proxies"))
    env.pop("PARALLEL_SEARCH_ENABLED", None)
    if value is not None:
        env["PARALLEL_SEARCH_ENABLED"] = value
    output = subprocess.check_output(
        [sys.executable, "-c", "import search_providers as sp; print(sp.PARALLEL_SEARCH_ENABLED)"], env=env,
    )
    assert output.decode().strip() == str(enabled)


@pytest.fixture
def parallel_server(monkeypatch):
    requests = []
    state = {
        "result": {
            "content": [],
            "structuredContent": {"results": [{
                "title": "Task groups", "url": "https://docs.python.org/3/library/asyncio-task.html",
                "excerpts": ["TaskGroup waits for its tasks."], "publish_date": "2026-01-01",
            }]},
        },
    }

    async def handle(request):
        # Observe actual SDK requests, including initialize and notifications.
        assert str(request.url) == "https://search.parallel.ai/mcp"
        assert request.headers["user-agent"] == (
            "deep-search-portal/parallel-search (+https://github.com/OrpingtonClose/deep-search-portal)"
        )
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        requests.append(body)
        method = body["method"]
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-search", "version": "1"},
            }
        elif method.startswith("notifications/"):
            return httpx.Response(202)
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            state["called"].set()
            if state.get("hang"):
                try:
                    await asyncio.Event().wait()
                finally:
                    state["cancelled"].set()
            if state.get("status"):
                return httpx.Response(state["status"], text="Service unavailable")
            result = state["result"]
        else:
            raise AssertionError(f"Unexpected MCP method {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def factory(headers=None, timeout=None, auth=None):
        assert timeout.connect == sp._PARALLEL_TIMEOUT
        assert timeout.read == sp._PARALLEL_TIMEOUT
        return httpx.AsyncClient(
            headers=headers, timeout=timeout, auth=auth, transport=httpx.MockTransport(handle),
        )

    original = streamable_http.streamablehttp_client

    def connect(*args, **kwargs):
        return original(*args, **kwargs, httpx_client_factory=factory)

    monkeypatch.setattr(streamable_http, "streamablehttp_client", connect)
    monkeypatch.setattr(sp, "PARALLEL_SEARCH_ENABLED", True)
    for provider in ("duckduckgo", "brave", "exa", "firecrawl", "mojeek", "searxng"):
        monkeypatch.setattr(sp, f"_search_{provider}", AsyncMock(return_value=[]))
    state["called"] = asyncio.Event()
    state["cancelled"] = asyncio.Event()
    return state, requests


@pytest.mark.asyncio
async def test_general_search_request_and_raw_output(parallel_server):
    _, requests = parallel_server
    results = await sp.search_as_raw("Python asyncio TaskGroup documentation")
    assert results == [{
        "title": "Task groups", "url": "https://docs.python.org/3/library/asyncio-task.html",
        "content": "TaskGroup waits for its tasks.", "source": "parallel", "publishedDate": "2026-01-01",
    }]
    calls = [r["params"] for r in requests if r["method"] == "tools/call"]
    assert calls == [{"name": "web_search", "arguments": {
        "objective": "Python asyncio TaskGroup documentation",
        "search_queries": ["Python asyncio TaskGroup documentation"],
    }}]
    assert requests[0]["method"] == "initialize"
    assert sp.available_providers()["parallel"] is True


@pytest.mark.asyncio
async def test_defaults_make_no_parallel_requests(parallel_server, monkeypatch):
    _, requests = parallel_server
    monkeypatch.setattr(sp, "PARALLEL_SEARCH_ENABLED", False)
    assert await sp.search_as_raw("test") == []
    assert await sp._search_parallel("test") == []
    assert sp.available_providers()["parallel"] is False
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("category,time_range", [
    ("general", "week"), ("news", ""), ("science", ""), ("videos", ""),
])
async def test_filtered_and_specialized_routes_unchanged(parallel_server, category, time_range):
    _, requests = parallel_server
    # News has a dedicated DuckDuckGo caller in addition to the general providers.
    with patch.object(sp, "_search_duckduckgo_news", AsyncMock(return_value=[])):
        await sp.search_as_raw("test", categories=category, time_range=time_range)
    assert requests == []


@pytest.mark.asyncio
async def test_merge_deduplication_priority_and_limit(parallel_server, monkeypatch):
    state, _ = parallel_server
    state["result"]["structuredContent"]["results"] += [
        {"title": "New source", "url": "https://example.org", "excerpts": ["New snippet"]},
    ]
    monkeypatch.setattr(sp, "_search_duckduckgo", AsyncMock(return_value=[sp.SearchResult(
        title="Existing result", url="https://docs.python.org/3/library/asyncio-task.html",
        snippet="Existing snippet", source="duckduckgo",
    )]))
    results = await sp.search_as_raw("test", max_results=2)
    assert [r["source"] for r in results] == ["duckduckgo", "parallel"]
    assert results[0]["title"] == "Existing result"
    assert results[1]["content"] == "New snippet"
    assert len(await sp.search_as_raw("test", max_results=1)) == 1


@pytest.mark.asyncio
async def test_text_json_nullable_fields_and_bad_rows(parallel_server):
    state, _ = parallel_server
    state["result"] = {"content": [{"type": "text", "text": json.dumps({"results": [
        None, {"url": "javascript:alert(1)"}, {"url": None},
        {"url": "https://example.org", "title": None, "publish_date": None,
         "excerpts": ["x" * 600, None]},
        {"url": "https://second.example.org", "excerpts": []},
    ]})}]}
    results = await sp.search_as_raw("test", max_results=1)
    assert results == [{
        "title": "https://example.org", "url": "https://example.org",
        "content": "x" * 500, "source": "parallel", "publishedDate": "",
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["tool", "http", "malformed"])
async def test_provider_failures_preserve_other_results(parallel_server, monkeypatch, failure):
    state, _ = parallel_server
    if failure == "tool":
        state["result"] = {"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]}
    elif failure == "http":
        state["status"] = 503
    else:
        state["result"] = {"content": [{"type": "text", "text": "invalid JSON"}]}
    monkeypatch.setattr(sp, "_search_duckduckgo", AsyncMock(return_value=[sp.SearchResult(
        title="Existing", url="https://example.org", snippet="Still available", source="duckduckgo",
    )]))
    results = await sp.search_as_raw("test")
    assert len(results) == 1
    assert results[0]["content"] == "Still available"


@pytest.mark.asyncio
async def test_total_timeout_cancels_request(parallel_server, monkeypatch):
    state, _ = parallel_server
    state["hang"] = True
    monkeypatch.setattr(sp, "_PARALLEL_TIMEOUT", 0.1)
    assert await asyncio.wait_for(sp.search_as_raw("test"), timeout=2) == []
    assert state["called"].is_set()
    assert state["cancelled"].is_set()


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_and_cleans_up(parallel_server):
    state, _ = parallel_server
    state["hang"] = True
    task = asyncio.create_task(sp.search_as_raw("test"))
    await asyncio.wait_for(state["called"].wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert state["cancelled"].is_set()
