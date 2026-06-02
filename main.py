import os
import sys

from dotenv import load_dotenv
from crewai import Crew, Process
from crewai_tools import ScrapeWebsiteTool
from agents import create_llm, create_agents
from tasks import create_tasks
from tools import RSSNewsSearchTool


class ConfigError(Exception):
    pass


def validate_env():
    missing = []
    for var in ("DEEPSEEK_API_KEY",):
        if not os.getenv(var):
            missing.append(var)
    if missing:
        raise ConfigError(
            f"缺少环境变量: {', '.join(missing)}。"
            f"请在 .env 文件中配置。参见 .env.example。"
        )


def run_crew(topic: str, max_articles: int = 5, days_back: int = 7):
    load_dotenv()
    validate_env()

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

    return news_crew.kickoff(inputs={
        "topic": topic,
        "max_articles": str(max_articles),
        "days_back": str(days_back),
    })
