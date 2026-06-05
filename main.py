import os
import re

from dotenv import load_dotenv
from crewai import Crew, Process, Task
from agents import create_llm, create_agents, create_auditor, create_fact_checker
from tasks import create_tasks, create_report_task, create_fact_check_task, create_audit_task
from tools import WeChatSearchTool


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


def _count_errors(fact_check_output: str) -> int:
    """Parse fact-check output to count discovered errors."""
    # count error sections: each "错误陈述" or numbered error = 1 error
    errors = re.findall(r"错误陈述", fact_check_output)
    # also count ❌ markers
    cross_marks = fact_check_output.count("❌")
    return max(len(errors), cross_marks)


def run_crew(topic: str, max_articles: int = 5, days_back: int = 7, provider: str = "siliconflow"):
    load_dotenv()
    validate_env(provider)

    llm = create_llm(provider)
    search_tool = WeChatSearchTool()
    inputs = {
        "topic": topic,
        "max_articles": str(max_articles),
        "days_back": str(days_back),
    }

    collector, analyzer, reporter = create_agents(search_tool, llm)
    collect_task, analyze_task = create_tasks(collector, analyzer)
    fact_checker = create_fact_checker(llm)
    auditor = create_auditor(llm)

    # ── Phase 1: collect → analyze → fact-check ────────────
    fact_check_task = create_fact_check_task(fact_checker, collect_task, analyze_task)
    phase1 = Crew(
        agents=[collector, analyzer, fact_checker],
        tasks=[collect_task, analyze_task, fact_check_task],
        process=Process.sequential,
        verbose=True,
    )
    result1 = phase1.kickoff(inputs=inputs)
    fc_output = result1.tasks_output[2].raw if len(result1.tasks_output) > 2 else ""

    # ── Phase 2: mandatory correction round #1 ──────────────
    report_task = create_report_task(reporter, collect_task, analyze_task, fact_check_task)

    fix_task = Task(
        description=(
            "你必须逐条修正以下事实核查发现的每一个错误，"
            "不允许跳过任何一条。每条修正后标注 ✅已修正。\n\n"
            "## 需要修正的错误（逐条处理，不可遗漏）\n"
            f"{fc_output}\n\n"
            "修正完成后，在报告末尾附上修正清单: 共修正 X 条。"
            "如果没有发现错误，直接输出原报告，标注 ✅无需修正。"
        ),
        expected_output="修正后的完整 Markdown 报告，含修正清单。",
        agent=reporter,
        context=[collect_task, analyze_task, report_task, fact_check_task],
    )
    recheck_task = Task(
        description=(
            "逐条核查以下错误是否在报告中被修正。输出核查结果:\n"
            "| 错误编号 | 错误简述 | 修正状态 | 备注 |\n"
            "|:---:|------|:---:|------|\n\n"
            f"{fc_output}\n\n"
            "如果仍有未修正的错误，在备注中说明具体问题。"
        ),
        expected_output="核查结果表: 每条错误的修正状态。最后一行: 剩余未修正错误数: X。",
        agent=fact_checker,
        context=[fix_task],
    )
    round1_crew = Crew(
        agents=[reporter, fact_checker],
        tasks=[fix_task, recheck_task],
        process=Process.sequential,
        verbose=True,
    )
    result2 = round1_crew.kickoff()
    report = result2.tasks_output[0].raw if len(result2.tasks_output) > 0 else ""
    fc2 = result2.tasks_output[1].raw if len(result2.tasks_output) > 1 else ""

    # ── Phase 2b: conditional second correction if ≥3 remain ──
    remaining = _count_errors(fc2)
    if remaining >= 3:
        fix_task2 = Task(
            description=(
                "上一轮还有以下错误未完全修正，必须逐条彻底修正:\n\n"
                f"{fc2}\n\n"
                "每条修正后标注 ✅已修正，不可跳过。"
            ),
            expected_output="最终修正后的完整 Markdown 报告。",
            agent=reporter,
            context=[fix_task, recheck_task],
        )
        fix2_crew = Crew(
            agents=[reporter],
            tasks=[fix_task2],
            process=Process.sequential,
            verbose=True,
        )
        result2b = fix2_crew.kickoff()
        report = result2b.tasks_output[0].raw if len(result2b.tasks_output) > 0 else report
        audited_report_task = fix_task2
    else:
        audited_report_task = fix_task

    # ── Phase 3: audit ─────────────────────────────────────
    audit_task = create_audit_task(
        auditor, collect_task, analyze_task, fact_check_task, audited_report_task
    )
    audit_crew = Crew(
        agents=[auditor],
        tasks=[audit_task],
        process=Process.sequential,
        verbose=True,
    )
    result3 = audit_crew.kickoff()
    audit = result3.tasks_output[0].raw if len(result3.tasks_output) > 0 else ""

    return report, audit
