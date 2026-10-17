"""
Multi-source search provider — replaces SearXNG as the sole search backend.

Instead of routing all searches through SearXNG (a meta-search engine that
proxies queries to Google/Bing/DuckDuckGo), this module queries multiple
independent sources directly:

  1. **DuckDuckGo** (via LangChain DuckDuckGoSearchAPIWrapper) — free, no key
  2. **Brave Search** (via LangChain BraveSearchWrapper) — independent index, needs API key
  3. **Mojeek** (via LangChain MojeekSearchAPIWrapper) — independent crawler, needs API key
  4. **SearXNG** — kept as fallback for categories not covered by direct sources
  5. **Exa** (via direct HTTP API) — semantic/neural AI search, needs API key
  6. **Firecrawl** (via direct HTTP API) — deep web scraping with search, needs API key
  7. **Parallel** (via Search MCP) — optional keyless general web search

Each provider returns normalised ``SearchResult`` dicts. The ``multi_search``
coroutine fans out to all available providers concurrently and deduplicates
by URL, producing a merged result set with provenance tags.

For specialised categories (news, science, videos), dedicated functions
query the appropriate providers directly rather than using generic web search.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from shared import get_throttler, http_client

log = logging.getLogger("search_providers")

# ---------------------------------------------------------------------------
# Configuration — provider availability determined by env vars
# ---------------------------------------------------------------------------

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
MOJEEK_API_KEY = os.getenv("MOJEEK_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

# Feature flag: set to "0" or "false" to disable SearXNG entirely
SEARXNG_ENABLED = os.getenv("SEARXNG_ENABLED", "1").lower() not in ("0", "false", "no")

# Opt in explicitly: existing search routing makes no Parallel requests by default.
PARALLEL_SEARCH_ENABLED = os.getenv("PARALLEL_SEARCH_ENABLED", "0").lower() in ("1", "true", "yes")
_PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
_PARALLEL_USER_AGENT = (
    "deep-search-portal/parallel-search (+https://github.com/OrpingtonClose/deep-search-portal)"
)
_PARALLEL_TIMEOUT = 20.0


# ---------------------------------------------------------------------------
# Normalised result type
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result from any provider."""
    title: str
    url: str
    snippet: str
    source: str  # provider name, e.g. "duckduckgo", "brave", "parallel", "searxng"
    score: float = 0.0  # optional relevance score
    published_date: str = ""  # ISO date if available
    img_src: str = ""         # Direct image URL (for image search results)
    thumbnail_src: str = ""   # Thumbnail URL (for image search results)

    def to_dict(self) -> dict:
        d = {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "score": self.score,
            "publishedDate": self.published_date,
        }
        if self.img_src:
            d["img_src"] = self.img_src
        if self.thumbnail_src:
            d["thumbnail_src"] = self.thumbnail_src
        return d


# ---------------------------------------------------------------------------
# Provider: Parallel (anonymous Streamable HTTP MCP)
# ---------------------------------------------------------------------------

async def _search_parallel(query: str, max_results: int = 10) -> list[SearchResult]:
    """Supplement general search with Parallel's opt-in, keyless Search MCP."""
    if not PARALLEL_SEARCH_ENABLED or max_results <= 0:
        return []

    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async def _call():
            async with streamablehttp_client(
                _PARALLEL_MCP_URL,
                headers={"User-Agent": _PARALLEL_USER_AGENT},
                timeout=_PARALLEL_TIMEOUT,
                sse_read_timeout=_PARALLEL_TIMEOUT,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(
                        "web_search", {"objective": query, "search_queries": [query]},
                    )

        # Bound the whole operation, including connection, initialization and cleanup.
        response = await asyncio.wait_for(_call(), timeout=_PARALLEL_TIMEOUT)
        if response.isError:
            log.warning("Parallel search returned a tool error")
            return []

        data = response.structuredContent
        if data is None:
            # Older MCP servers may expose the JSON only in text content.
            for block in response.content:
                if block.type == "text":
                    try:
                        candidate = json.loads(block.text)
                    except ValueError:
                        continue
                    if isinstance(candidate, dict) and "results" in candidate:
                        data = candidate
                        break
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValueError("Parallel response has no results list")

        results = []
        for item in data["results"]:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                continue
            excerpts = item.get("excerpts")
            snippet = ""
            if isinstance(excerpts, list):
                snippet = "\n".join(text for text in excerpts if isinstance(text, str))
            published_date = item.get("publish_date")
            results.append(SearchResult(
                title=item.get("title") if isinstance(item.get("title"), str) else url,
                url=url,
                snippet=snippet[:500],
                source="parallel",
                published_date=published_date if isinstance(published_date, str) else "",
            ))
            if len(results) >= max_results:
                break
        return results

    except Exception as e:
        # Match the other providers' fail-open behavior. CancelledError propagates.
        log.warning(f"Parallel search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: DuckDuckGo (via LangChain)
# ---------------------------------------------------------------------------

async def _search_duckduckgo(query: str, max_results: int = 10) -> list[SearchResult]:
    """Search DuckDuckGo using LangChain's DuckDuckGoSearchAPIWrapper.

    Free, no API key required. Uses the DDGS library under the hood.
    """
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

        ddg = DuckDuckGoSearchAPIWrapper(max_results=max_results)
        # Run sync wrapper in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, ddg.results, query, max_results)

        results = []
        for item in raw[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", "")[:500],
                source="duckduckgo",
            ))
        return results

    except ImportError:
        log.warning("DuckDuckGo search unavailable: langchain-community or duckduckgo-search not installed")
        return []
    except Exception as e:
        log.warning(f"DuckDuckGo search error: {e}")
        return []


async def _search_duckduckgo_news(query: str, max_results: int = 10) -> list[SearchResult]:
    """Search DuckDuckGo News using the DDGS library directly."""
    try:
        from duckduckgo_search import DDGS

        loop = asyncio.get_running_loop()

        def _run():
            with DDGS() as ddgs:
                return list(ddgs.news(query, max_results=max_results))

        raw = await loop.run_in_executor(None, _run)

        results = []
        for item in raw[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("body", "")[:500],
                source="duckduckgo_news",
                published_date=item.get("date", ""),
            ))
        return results

    except ImportError:
        log.debug("DuckDuckGo news unavailable: duckduckgo-search not installed")
        return []
    except Exception as e:
        log.warning(f"DuckDuckGo news error: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: Brave Search (via LangChain)
# ---------------------------------------------------------------------------

async def _search_brave(query: str, max_results: int = 10) -> list[SearchResult]:
    """Search Brave using LangChain's BraveSearchWrapper.

    Requires BRAVE_SEARCH_API_KEY. Brave has its own independent web index.
    """
    if not BRAVE_SEARCH_API_KEY:
        return []

    try:
        from langchain_community.utilities import BraveSearchWrapper

        brave = BraveSearchWrapper(
            api_key=BRAVE_SEARCH_API_KEY,
            search_kwargs={"count": max_results},
        )
        loop = asyncio.get_running_loop()
        raw_json = await loop.run_in_executor(None, brave.run, query)

        # BraveSearchWrapper returns a JSON string of results
        try:
            items = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            # Sometimes returns plain text — wrap it
            if raw_json and isinstance(raw_json, str):
                return [SearchResult(
                    title="Brave Search Result",
                    url="",
                    snippet=raw_json[:500],
                    source="brave",
                )]
            return []

        results = []
        if isinstance(items, list):
            for item in items[:max_results]:
                if isinstance(item, dict):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", item.get("url", "")),
                        snippet=item.get("snippet", item.get("description", ""))[:500],
                        source="brave",
                    ))
        return results

    except ImportError:
        log.warning("Brave search unavailable: langchain-community not installed")
        return []
    except Exception as e:
        log.warning(f"Brave search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: Mojeek (via LangChain)
# ---------------------------------------------------------------------------

async def _search_mojeek(query: str, max_results: int = 10) -> list[SearchResult]:
    """Search Mojeek using LangChain's MojeekSearchAPIWrapper.

    Requires MOJEEK_API_KEY. Mojeek has its own independent crawler — no
    Google/Bing proxy, making it valuable for diversity.
    """
    if not MOJEEK_API_KEY:
        return []

    try:
        from langchain_community.utilities import MojeekSearchAPIWrapper

        mojeek = MojeekSearchAPIWrapper(api_key=MOJEEK_API_KEY)
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, mojeek.results, query, max_results)

        results = []
        for item in raw[:max_results]:
            if isinstance(item, dict):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", item.get("link", "")),
                    snippet=item.get("desc", item.get("snippet", ""))[:500],
                    source="mojeek",
                ))
        return results

    except ImportError:
        log.warning("Mojeek search unavailable: langchain-community not installed")
        return []
    except Exception as e:
        log.warning(f"Mojeek search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: Exa (semantic AI search — direct HTTP)
# ---------------------------------------------------------------------------

async def _search_exa(query: str, max_results: int = 10) -> list[SearchResult]:
    """Search via Exa's semantic/neural search API.

    Requires EXA_API_KEY. Exa uses neural embeddings to find conceptually
    relevant results — excels at research, technical content, and niche topics.
    """
    if not EXA_API_KEY:
        return []

    try:
        client = http_client()
        resp = await client.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": EXA_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "numResults": max_results,
                "type": "auto",
                "useAutoprompt": True,
                "text": True,
            },
            timeout=20.0,
        )
        if resp.status_code != 200:
            log.warning(f"Exa HTTP {resp.status_code}: {resp.text[:200]}")
            return []

        data = resp.json()
        results = []
        for item in data.get("results", [])[:max_results]:
            text = item.get("text", "")
            snippet = text[:500] if text else item.get("highlight", "")[:500]
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=snippet,
                source="exa",
                score=item.get("score", 0.0),
                published_date=item.get("publishedDate", ""),
            ))
        return results

    except Exception as e:
        log.warning(f"Exa search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: Firecrawl (deep web scraping with search)
# ---------------------------------------------------------------------------

async def _search_firecrawl(query: str, max_results: int = 10) -> list[SearchResult]:
    """Search via Firecrawl's /search endpoint.

    Requires FIRECRAWL_API_KEY. Firecrawl searches the web and returns
    clean markdown content — handles JS rendering, anti-bot, CAPTCHAs.
    Unlike basic search, returns actual page content, not just snippets.
    """
    if not FIRECRAWL_API_KEY:
        return []

    try:
        client = http_client()
        resp = await client.post(
            "https://api.firecrawl.dev/v1/search",
            headers={
                "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "limit": max_results,
                "scrapeOptions": {"formats": ["markdown"]},
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            log.warning(f"Firecrawl HTTP {resp.status_code}: {resp.text[:200]}")
            return []

        data = resp.json()
        results = []
        for item in data.get("data", [])[:max_results]:
            metadata = item.get("metadata", {})
            # Firecrawl returns full markdown — use first 500 chars as snippet
            markdown = item.get("markdown", "")
            snippet = markdown[:500] if markdown else metadata.get("description", "")[:500]
            results.append(SearchResult(
                title=metadata.get("title", item.get("url", "")),
                url=item.get("url", ""),
                snippet=snippet,
                source="firecrawl",
                published_date=metadata.get("publishedDate", ""),
            ))
        return results

    except Exception as e:
        log.warning(f"Firecrawl search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: SearXNG (direct HTTP, kept as fallback)
# ---------------------------------------------------------------------------

async def _search_searxng(
    query: str,
    categories: str = "general",
    time_range: str = "",
    max_results: int = 10,
) -> list[SearchResult]:
    """Query SearXNG instance and return normalised results.

    This is the legacy path — kept as fallback when direct providers
    are unavailable or for categories they don't cover (e.g. onions, files).
    """
    if not SEARXNG_ENABLED:
        return []

    try:
        async with get_throttler("searxng").throttle():
            client = http_client()
            params: dict[str, str] = {
                "q": query,
                "format": "json",
                "categories": categories,
            }
            if time_range:
                params["time_range"] = time_range

            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params=params,
                timeout=20.0,
            )
            if resp.status_code != 200:
                log.warning(f"SearXNG HTTP {resp.status_code} for categories={categories}")
                return []

            data = resp.json()
            raw_results = data.get("results", [])[:max_results]

            results = []
            for r in raw_results:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", "")[:500],
                    source=f"searxng:{categories}",
                    published_date=r.get("publishedDate", ""),
                    img_src=r.get("img_src", ""),
                    thumbnail_src=r.get("thumbnail_src", ""),
                ))
            return results

    except httpx.TimeoutException:
        log.warning("SearXNG timeout")
        return []
    except Exception as e:
        log.warning(f"SearXNG error: {e}")
        return []


# ---------------------------------------------------------------------------
# Deduplication and merging
# ---------------------------------------------------------------------------

def _normalise_url(url: str) -> str:
    """Normalise a URL for dedup: strip protocol, trailing slash, www."""
    url = url.strip()
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
    if url.startswith("www."):
        url = url[4:]
    return url.rstrip("/")


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate URLs, preserving first occurrence (priority order)."""
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for r in results:
        norm = _normalise_url(r.url)
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Public API: multi-source search
# ---------------------------------------------------------------------------

async def multi_search(
    query: str,
    max_results: int = 10,
    include_news: bool = False,
    time_range: str = "",
) -> list[SearchResult]:
    """Fan out to all available search providers and merge results.

    Provider priority (results listed first have higher placement):
      1. DuckDuckGo (always available, free)
      2. Brave (if API key configured)
      3. Exa, Firecrawl, Mojeek (if API keys configured)
      4. SearXNG (if enabled)
      5. Parallel (if explicitly enabled, without a time_range)

    Results are deduplicated by normalised URL.
    """
    tasks = [
        _search_duckduckgo(query, max_results),
        _search_brave(query, max_results),
        _search_exa(query, max_results),
        _search_firecrawl(query, max_results),
        _search_mojeek(query, max_results),
        _search_searxng(query, categories="general", time_range=time_range, max_results=max_results),
    ]

    if include_news:
        tasks.append(_search_duckduckgo_news(query, max_results))

    # MCP has no time-range filter; keep time-filtered searches on existing routes.
    if PARALLEL_SEARCH_ENABLED and not time_range:
        tasks.append(_search_parallel(query, max_results))

    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: list[SearchResult] = []
    for batch in gathered:
        if isinstance(batch, list):
            all_results.extend(batch)
        elif isinstance(batch, Exception):
            log.warning(f"Search provider error: {batch}")

    return _deduplicate(all_results)[:max_results]


async def multi_search_news(
    query: str,
    time_range: str = "week",
    max_results: int = 10,
) -> list[SearchResult]:
    """Search for recent news across multiple providers.

    Priority:
      1. DuckDuckGo News (free, good recency)
      2. SearXNG news category (fallback)
      3. SearXNG general with time_range (supplement)
    """
    tasks = [
        _search_duckduckgo_news(query, max_results),
        _search_searxng(query, categories="news", time_range=time_range, max_results=max_results),
    ]

    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: list[SearchResult] = []
    for batch in gathered:
        if isinstance(batch, list):
            all_results.extend(batch)
        elif isinstance(batch, Exception):
            log.warning(f"News search provider error: {batch}")

    # Supplement with general search for additional coverage
    if len(all_results) < 3:
        try:
            general = await _search_searxng(
                query, categories="general", time_range=time_range, max_results=5,
            )
            all_results.extend(general)
        except Exception:
            pass

    return _deduplicate(all_results)[:max_results]


async def multi_search_science(
    query: str,
    max_results: int = 10,
) -> list[SearchResult]:
    """Search academic/science sources across multiple providers.

    Priority:
      1. SearXNG science category (aggregates Google Scholar, Semantic Scholar)
      2. DuckDuckGo with academic site targeting (fallback)
    """
    tasks = [
        _search_exa(query, max_results),  # Exa excels at research/academic content
        _search_searxng(query, categories="science", max_results=max_results),
    ]

    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: list[SearchResult] = []
    for batch in gathered:
        if isinstance(batch, list):
            all_results.extend(batch)
        elif isinstance(batch, Exception):
            log.warning(f"Science search provider error: {batch}")

    # If SearXNG science returned nothing, use DuckDuckGo with academic site targeting
    if not all_results:
        academic_query = (
            f"({query}) (site:scholar.google.com OR site:semanticscholar.org "
            f"OR site:researchgate.net OR site:academia.edu OR site:ssrn.com "
            f"OR site:jstor.org OR site:ncbi.nlm.nih.gov)"
        )
        try:
            fallback = await _search_duckduckgo(academic_query, max_results)
            all_results.extend(fallback)
        except Exception:
            pass

    return _deduplicate(all_results)[:max_results]


async def multi_search_site(
    query: str,
    site: str,
    max_results: int = 10,
) -> list[SearchResult]:
    """Search a specific site across multiple providers.

    Used for site-targeted searches (e.g. site:substack.com, site:reddit.com).
    """
    site_query = f"site:{site} {query}"

    tasks = [
        _search_duckduckgo(site_query, max_results),
        _search_brave(site_query, max_results),
        _search_exa(site_query, max_results),
        _search_firecrawl(site_query, max_results),
        _search_searxng(site_query, categories="general", max_results=max_results),
    ]

    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: list[SearchResult] = []
    for batch in gathered:
        if isinstance(batch, list):
            all_results.extend(batch)
        elif isinstance(batch, Exception):
            log.warning(f"Site search provider error: {batch}")

    return _deduplicate(all_results)[:max_results]


async def multi_search_forums(
    query: str,
    forum_sites: list[str],
    max_results: int = 15,
) -> list[SearchResult]:
    """Search across multiple forum sites concurrently.

    Fans out site-targeted searches to all available providers for each
    batch of forum sites, plus a generic forum query.
    """
    tasks: list = []

    # Batch forum sites into groups of 5 for site-targeted search
    batch_size = 5
    for i in range(0, min(len(forum_sites), 15), batch_size):
        batch = forum_sites[i:i + batch_size]
        site_clause = " OR ".join(f"site:{s}" for s in batch)
        combined_query = f"({site_clause}) {query}"
        tasks.append(_search_duckduckgo(combined_query, max_results=5))
        tasks.append(_search_searxng(combined_query, categories="general", max_results=5))

    # Generic forum query
    tasks.append(_search_duckduckgo(f"{query} forum discussion thread", max_results=5))

    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: list[SearchResult] = []
    for batch in gathered:
        if isinstance(batch, list):
            all_results.extend(batch)
        elif isinstance(batch, Exception):
            log.warning(f"Forum search provider error: {batch}")

    return _deduplicate(all_results)[:max_results]


# ---------------------------------------------------------------------------
# Raw result conversion (backward compatibility with _searxng_query callers)
# ---------------------------------------------------------------------------

def results_to_raw_dicts(results: list[SearchResult]) -> list[dict]:
    """Convert SearchResult list to raw dicts matching SearXNG's format.

    This allows callers that expect the old _searxng_query return format
    to work without modification during the transition.
    """
    return [
        {
            "title": r.title,
            "url": r.url,
            "content": r.snippet,
            "source": r.source,
            "publishedDate": r.published_date,
        }
        for r in results
    ]


async def search_as_raw(
    query: str,
    categories: str = "general",
    time_range: str = "",
    max_results: int = 10,
) -> list[dict]:
    """Drop-in replacement for _searxng_query that routes to multi-source.

    Returns raw dicts in the same format as SearXNG results, so existing
    callers (tool_searxng_search, tool_forum_search, etc.) work unchanged.

    Category routing:
      - "general" -> multi_search
      - "news"    -> multi_search_news
      - "science" -> multi_search_science
      - "videos"  -> SearXNG fallback (videos handled by YouTube tools)
      - other     -> SearXNG fallback
    """
    if categories == "general":
        results = await multi_search(query, max_results=max_results, time_range=time_range)
    elif categories == "news":
        results = await multi_search_news(query, time_range=time_range, max_results=max_results)
    elif categories == "science":
        results = await multi_search_science(query, max_results=max_results)
    else:
        # For uncovered categories (videos, files, onions, etc.), fall back to SearXNG
        results = await _search_searxng(
            query, categories=categories, time_range=time_range, max_results=max_results,
        )

    return results_to_raw_dicts(results)


# ---------------------------------------------------------------------------
# Provider availability check
# ---------------------------------------------------------------------------

def available_providers() -> dict[str, bool]:
    """Return which search providers are configured and available."""
    ddg_available = True
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper  # noqa: F401
    except ImportError:
        ddg_available = False

    return {
        "duckduckgo": ddg_available,
        "brave": bool(BRAVE_SEARCH_API_KEY),
        "exa": bool(EXA_API_KEY),
        "firecrawl": bool(FIRECRAWL_API_KEY),
        "mojeek": bool(MOJEEK_API_KEY),
        "searxng": SEARXNG_ENABLED,
        "parallel": PARALLEL_SEARCH_ENABLED,
    }
