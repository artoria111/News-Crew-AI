import os
import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool

# quality filter — skip low-substance articles (shared by both search tools)
_SPAM_KEYWORDS = [
    "讲座预告",
    "讲座通知",
    "活动预告",
    "活动报名",
    "会议通知",
    "报名开始",
    "扫码报名",
    "免费领取",
    "限时优惠",
    "领券",
    "招聘公告",
    "招聘启事",
    "培训通知",
    "开班通知",
    "直播预告",
    "直播预约",
    "今晚直播",
    "有奖转发",
    "转发抽奖",
    "投票",
    "停水通知",
    "停电通知",
    "放假通知",
    "天气周报",
    "一文读懂",
    "大赛报名",
    "学院",
]


class WeChatSearchTool(BaseTool):
    name: str = "WeChat News Search"
    description: str = (
        "Search WeChat public account articles via Sogou. "
        "Input: a search query in Chinese. "
        "Returns: matched articles from diverse WeChat accounts "
        "with title, URL, source account, date, and snippet."
    )

    @staticmethod
    def _clean(text: str) -> str:
        # remove surrogate characters that break llm encoding
        return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    def _run(self, query: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        results = []

        for attempt in range(2):
            try:
                resp = requests.get(
                    "https://weixin.sogou.com/weixin",
                    params={"type": 2, "query": query},
                    headers=headers,
                    timeout=15,
                )
                resp.encoding = "utf-8"
            except requests.RequestException as e:
                if attempt == 1:
                    return f"WeChat search request failed: {e}"
                time.sleep(2)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            items = soup.select(".txt-box")
            if items:
                break  # got results, proceed
            time.sleep(2)  # empty page — retry

        for item in items:
            # title + link
            title_el = item.select_one("h3 a")
            if not title_el:
                continue
            title = self._clean(title_el.get_text(strip=True))
            link = title_el.get("href", "")
            if link.startswith("/"):
                link = "https://weixin.sogou.com" + link

            # source account
            source_el = item.select_one(".s-p .all-time-y2")
            source = (
                self._clean(source_el.get_text(strip=True))
                if source_el
                else "未知公众号"
            )

            # date from JS timestamp — filter out articles older than 30 days
            date = ""
            s2_el = item.select_one(".s-p .s2")
            if s2_el:
                script_el = s2_el.select_one("script")
                script_text = script_el.get_text(strip=True) if script_el else ""
                ts_match = re.search(r"timeConvert\('(\d+)'\)", script_text)
                if ts_match:
                    from datetime import datetime, timedelta

                    try:
                        ts = int(ts_match.group(1))
                        # handle millisecond timestamps
                        if ts > 1e12:
                            ts = ts // 1000
                        dt = datetime.fromtimestamp(ts)
                        # sanity check: reject dates before 2025 or after 2030
                        if dt.year < 2025 or dt.year > 2030:
                            continue
                        # reject articles older than 30 days
                        if (datetime.now() - dt) > timedelta(days=30):
                            continue
                        date = dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass

            # snippet
            snippet_el = item.select_one(".txt-info")
            snippet = self._clean(snippet_el.get_text(strip=True)) if snippet_el else ""
            snippet = re.sub(r"^·\s*", "", snippet)[:300]

            # quality filter — skip low-substance articles
            check_text = title + " " + snippet
            if any(kw in check_text for kw in _SPAM_KEYWORDS):
                continue

            results.append(
                f"Title: {title}\n"
                f"URL: {link}\n"
                f"Source: {source} (微信公众号)\n"
                f"Date: {date}\n"
                f"Snippet: {snippet}\n"
            )

        if not results:
            return f"No WeChat articles found for: {query}"

        return "\n---\n".join(results[:15])


class BochaSearchTool(BaseTool):
    name: str = "Bocha Web Search"
    description: str = (
        "Search the entire Chinese web via the Bocha search API. "
        "Input: a search query in Chinese. "
        "Returns: matched web articles from diverse sites "
        "with title, URL, source site, date, and summary."
    )
    days_back: int = 7
    max_results: int = 10

    @staticmethod
    def _freshness(days: int) -> str:
        if days <= 1:
            return "oneDay"
        if days <= 7:
            return "oneWeek"
        if days <= 31:
            return "oneMonth"
        return "oneYear"

    @staticmethod
    def _clean(text: str) -> str:
        # remove surrogate characters that break llm encoding
        return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    def _run(self, query: str) -> str:
        api_key = os.getenv("BOCHA_API_KEY")
        if not api_key:
            return "Bocha API key missing: 请在 .env 中配置 BOCHA_API_KEY"

        base = os.getenv("BOCHA_BASE_URL", "https://api.bochaai.com/v1").rstrip("/")
        url = f"{base}/web-search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "count": self.max_results,
            "freshness": self._freshness(self.days_back),
            "summary": True,
        }

        resp = None
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 1:
                    return f"Bocha search request failed: {e}"
                time.sleep(2)

        body = resp.json() or {}
        if body.get("code") not in (None, 200):
            return f"Bocha search API error {body.get('code')}: {body.get('msg')}"

        webpages = ((body.get("data") or {}).get("webPages") or {}).get("value") or []
        results = []
        now = datetime.now()
        # bocha's datePublished is "page update time", not "event time".
        # to filter out retrospective/summary articles (e.g. "2026年3月19日 AI 回顾"),
        # extract any explicit date mentioned in title/snippet and reject if it is
        # more than 90 days older than the page update date.
        _DATE_PATTERNS = [
            re.compile(
                r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})"
            ),  # 2026-03-19 / 2026/3/19
            re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日"),  # 2026年3月19日
        ]

        def _extract_mentioned_dates(text: str) -> list:
            found = []
            for pat in _DATE_PATTERNS:
                for m in pat.finditer(text):
                    try:
                        if m.lastindex and m.lastindex >= 3:
                            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                            found.append(datetime(y, mo, d))
                    except ValueError:
                        pass
            return found

        for item in webpages:
            title = self._clean(item.get("name") or "")
            link = item.get("url") or ""
            if not title or not link:
                continue

            site = item.get("siteName") or "未知来源"
            snippet = self._clean(item.get("summary") or item.get("snippet") or "")

            # date from datePublished / dateLastCrawled (UTC+8 per Bocha docs)
            date = ""
            page_dt = None
            for field in ("datePublished", "dateLastCrawled"):
                raw = item.get(field)
                if raw:
                    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(raw))
                    if m:
                        date = m.group(1)
                        try:
                            page_dt = datetime.strptime(date, "%Y-%m-%d")
                        except ValueError:
                            page_dt = None
                        break

            # sanity check: reject impossible dates or articles out of range
            if page_dt:
                if page_dt.year < 2025 or page_dt.year > 2030:
                    continue
                if (now - page_dt) > timedelta(days=self.days_back + 1):
                    continue

            # retrospective / summary filter: if title/snippet explicitly mentions
            # a date that is much older than the page update, treat as old news.
            check_text = title + " " + snippet
            mentioned = _extract_mentioned_dates(check_text)
            if page_dt:
                oldest = min(mentioned) if mentioned else None
                if oldest and (page_dt - oldest) > timedelta(days=90):
                    continue

            # quality filter — skip low-substance articles
            if any(kw in check_text for kw in _SPAM_KEYWORDS):
                continue

            results.append(
                f"Title: {title}\n"
                f"URL: {link}\n"
                f"Source: {site}\n"
                f"Date: {date or '未知'}\n"
                f"Snippet: {snippet[:300]}\n"
            )

        if not results:
            return f"No web articles found for: {query}"

        return "\n---\n".join(results[: self.max_results])
