import os
import sys
from dotenv import load_dotenv
from crewai import Crew, Process
from crewai_tools import ScrapeWebsiteTool
from agents import create_llm, create_agents
from tasks import create_tasks
from tools import RSSNewsSearchTool


def validate_env():
    missing = []
    for var in ("DEEPSEEK_API_KEY",):
        if not os.getenv(var):
            missing.append(var)
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("Please set them in the .env file. See .env for reference.")
        sys.exit(1)


def main():
    load_dotenv()
    validate_env()

    print("=" * 60)
    print("  News Crew AI - 新闻搜集分析系统")
    print("=" * 60)

    topic = input("请输入要搜索的新闻主题 (默认: AI最新进展): ").strip()
    if not topic:
        topic = "AI最新进展"

    max_articles = input("搜集文章数量 (默认:  5): ").strip()
    if not max_articles:
        max_articles = "5"

    days_back = input("搜索时间范围（天）(默认:  7): ").strip()
    if not days_back:
        days_back = "7"

    print(f"\n开始搜集关于「{topic}」的新闻...")

    llm = create_llm()

    search_tool = RSSNewsSearchTool()
    scrape_tool = ScrapeWebsiteTool()

    collector, analyzer, reporter = create_agents(search_tool, scrape_tool, llm)
    collect_task, analyze_task, report_task = create_tasks(collector, analyzer, reporter)

    news_crew = Crew(
        agents=[collector, analyzer, reporter],
        tasks=[collect_task, analyze_task, report_task],
        process=Process.sequential,
        verbose=True,
    )

    result = news_crew.kickoff(inputs={
        "topic": topic,
        "max_articles": max_articles,
        "days_back": days_back,
    })

    print("\n" + "=" * 60)
    print("  最终报告")
    print("=" * 60 + "\n")
    print(result.raw)


if __name__ == "__main__":
    main()
