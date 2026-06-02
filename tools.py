import requests
import feedparser
from crewai.tools import BaseTool

NEWS_FEEDS = [
    {
        "name": "36氪",
        "url": "https://36kr.com/feed",
    },
    {
        "name": "澎湃新闻",
        "url": "https://www.thepaper.cn/rss_www.xml",
    },
]


class RSSNewsSearchTool(BaseTool):
    name: str = "RSS News Search"
    description: str = (
        "Search news articles from Chinese news RSS feeds. "
        "Input: a search keyword or topic in Chinese. "
        "Returns: matched articles with title, URL, source, date, and summary."
    )

    def _fetch_all(self):
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        entries = []
        for feed in NEWS_FEEDS:
            try:
                resp = requests.get(feed["url"], headers=headers, timeout=20)
                resp.encoding = "utf-8"
                parsed = feedparser.parse(resp.text)
                for entry in parsed.entries:
                    entry["_source_name"] = feed["name"]
                    entries.append(entry)
            except Exception:
                continue
        return entries

    def _run(self, query: str) -> str:
        entries = self._fetch_all()
        if not entries:
            return "Failed to fetch news from any RSS feed."

        keywords = query.lower().split()
        matched = []
        for entry in entries:
            title = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            text = (title + " " + summary).lower()
            if any(kw in text for kw in keywords):
                matched.append(entry)

        for entry in entries:
            if entry not in matched:
                title = entry.get("title", "").lower()
                summary = entry.get("summary", entry.get("description", "")).lower()
                text = title + " " + summary
                if any(kw in text for kw in keywords):
                    matched.append(entry)

        if not matched:
            return f"No articles matching '{query}' found in {len(entries)} total articles."

        results = []
        for entry in matched[:15]:
            title = entry.get("title", "N/A")
            url = entry.get("link", "N/A")
            source = entry.get("_source_name", "N/A")
            date = entry.get("published", entry.get("updated", "N/A"))
            summary = entry.get("summary", entry.get("description", "N/A"))
            # Strip HTML tags from summary
            import re
            summary = re.sub(r"<[^>]+>", "", summary)[:300]

            results.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Source: {source}\n"
                f"Date: {date}\n"
                f"Summary: {summary}\n"
            )

        return "\n---\n".join(results)
