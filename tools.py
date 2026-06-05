import re
import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool


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

        try:
            resp = requests.get(
                "https://weixin.sogou.com/weixin",
                params={"type": 2, "query": query},
                headers=headers,
                timeout=15,
            )
            resp.encoding = "utf-8"
        except requests.RequestException as e:
            return f"WeChat search request failed: {e}"

        soup = BeautifulSoup(resp.text, "html.parser")

        for item in soup.select(".txt-box"):
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
            source = self._clean(source_el.get_text(strip=True)) if source_el else "未知公众号"

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
            spam_keywords = [
                "讲座预告", "讲座通知", "活动预告", "活动报名", "会议通知",
                "报名开始", "扫码报名", "免费领取", "限时优惠", "领券",
                "招聘公告", "招聘启事", "培训通知", "开班通知",
                "直播预告", "直播预约", "今晚直播",
                "有奖转发", "转发抽奖", "投票",
                "停水通知", "停电通知", "放假通知", "天气周报","一文读懂"
            ]
            check_text = title + " " + snippet
            if any(kw in check_text for kw in spam_keywords):
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
