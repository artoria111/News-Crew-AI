import os

from dotenv import load_dotenv
from crewai import Crew, Process
from agents import create_llm, create_agents, create_auditor
from tasks import create_tasks, create_audit_task
from tools import RSSNewsSearchTool


class ConfigError(Exception):
    pass


def validate_env(provider: str):
    required = {
        "deepseek": ["DEEPSEEK_API_KEY", "DEEPSEEK_MODEL_NAME"],
        "siliconflow": ["SILICONFLOW_API_KEY", "SILICONFLOW_MODEL_NAME"],
    }
    keys = required.get(provider, required["siliconflow"])
    missing = [v for v in keys if not os.getenv(v)]
    if missing:
        raise ConfigError(
            f"缺少环境变量 ({provider}): {', '.join(missing)}。"
            f"请在 .env 文件中配置。"
        )


def run_crew(topic: str, max_articles: int = 5, days_back: int = 7, provider: str = "siliconflow"):
    load_dotenv()
    validate_env(provider)

    llm = create_llm(provider)
    search_tool = RSSNewsSearchTool()

    collector, analyzer, reporter = create_agents(search_tool, llm)
    collect_task, analyze_task, report_task = create_tasks(collector, analyzer, reporter)

    auditor = create_auditor(llm)
    audit_task = create_audit_task(auditor, collect_task, analyze_task, report_task)

    news_crew = Crew(
        agents=[collector, analyzer, reporter, auditor],
        tasks=[collect_task, analyze_task, report_task, audit_task],
        process=Process.sequential,
        verbose=True,
    )

    result = news_crew.kickoff(inputs={
        "topic": topic,
        "max_articles": str(max_articles),
        "days_back": str(days_back),
    })

    tasks_output = result.tasks_output
    report = tasks_output[2].raw if len(tasks_output) > 2 else ""
    audit = tasks_output[3].raw if len(tasks_output) > 3 else ""

    return report, audit
