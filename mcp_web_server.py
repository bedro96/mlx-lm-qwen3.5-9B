"""
MCP stdio server providing web browsing capabilities via Playwright.

Tools:
  - web_search : Search the web using DuckDuckGo and return the top results.
  - fetch_page : Fetch a web page and return its visible text content.

Usage (standalone test):
    uv run mcp_web_server.py

In production it is launched as a subprocess by main_mcp.py alongside
mcp_server.py so the LLM can call both factory sensor tools and web tools.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright

mcp = FastMCP("web-browser")

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Limit returned content to avoid flooding the LLM context window.
_MAX_CONTENT_CHARS = 8_000


def _clean_text(text: str) -> str:
    """Collapse excessive blank lines and spaces from extracted page text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web using Bing and return the top results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (1–10).
    """
    max_results = max(1, min(max_results, 10))
    search_url = f"https://www.bing.com/search?q={quote_plus(query)}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=_USER_AGENT)
        page = await context.new_page()
        try:
            await page.goto(search_url, wait_until="load", timeout=25_000)
            await page.wait_for_selector("#b_results", timeout=10_000)
            results: list[dict[str, str]] = await page.evaluate(
                """
                function() {
                    var items = [];
                    var lis = document.querySelectorAll("#b_results > li.b_algo");
                    for (var i = 0; i < lis.length; i++) {
                        var li   = lis[i];
                        var a    = li.querySelector("h2 > a");
                        var cite = li.querySelector("cite");
                        var cap  = li.querySelector(".b_caption p") || li.querySelector("p");
                        // cite text may contain breadcrumbs like "site.com › path › page"
                        var citeText = cite ? cite.textContent.trim() : "";
                        var url = citeText.replace(/\\s+[\\u203a>\\xbb]\\s+.*/g, "").trim();
                        if (url && !url.startsWith("http")) url = "https://" + url;
                        if (!url && a) url = a.href;
                        items.push({
                            title:   a   ? a.textContent.trim()   : "",
                            url:     url,
                            snippet: cap ? cap.textContent.trim() : ""
                        });
                    }
                    return items;
                }
                """
            )
        finally:
            await browser.close()

    trimmed = results[:max_results]
    return {
        "query": query,
        "results": trimmed,
        "count": len(trimmed),
    }


@mcp.tool()
async def fetch_page(url: str) -> dict[str, Any]:
    """Fetch a web page and return its visible text content.

    Args:
        url: The full URL to fetch (must start with http:// or https://).
    """
    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://", "url": url}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=_USER_AGENT)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            title: str = await page.title()
            text: str = await page.evaluate("() => document.body.innerText")
        finally:
            await browser.close()

    text = _clean_text(text)
    truncated = len(text) > _MAX_CONTENT_CHARS
    return {
        "url": url,
        "title": title,
        "content": text[:_MAX_CONTENT_CHARS],
        "truncated": truncated,
        "char_count": len(text),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
