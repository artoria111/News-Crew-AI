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
    errors = re.findall(r"错误陈述", fact_check_output)
    cross_marks = fact_check_output.count("❌")
    return max(len(errors), cross_marks)


def _validate_report(report: str, query: str) -> str:
    """Post-process: fix placeholder URLs and append real reference list if needed."""
    has_placeholder = bool(re.search(r"example\.com|example\.org|placeholder", report, re.I))
    if not has_placeholder:
        return report

    # Re-run search to get real URLs
    try:
        tool = WeChatSearchTool()
        raw = tool._run(query)
    except Exception:
        return report + "\n\n> ⚠️ URL 验证失败，请检查参考来源。"

    # Extract real URLs from tool output
    real_refs = []
    for part in raw.split("---"):
        t = re.search(r"Title:\s*(.+)", part)
        u = re.search(r"URL:\s*(.+)", part)
        s = re.search(r"Source:\s*(.+)", part)
        if t and u:
            real_refs.append(
                f"- [{t.group(1).strip()}]({u.group(1).strip()})"
                f" — {s.group(1).strip() if s else '未知来源'}"
            )

    if not real_refs:
        return report

    ref_section = "\n\n---\n\n## 📎 系统自动补充的参考来源\n\n" + "\n".join(real_refs)

    # Replace example.com URLs with real ones in the reference section
    report = re.sub(
        r"https?://example\.com[^\s\)\]\"]*",
        real_refs[0].split("](")[1].split(")")[0] if real_refs else "URL缺失",
        report,
    )

    return report + ref_section


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

    # ── Check: abort if no articles were collected ──────────
    collect_output = result1.tasks_output[0].raw if len(result1.tasks_output) > 0 else ""
    if re.search(r"共收集\s*0\s*篇|No WeChat articles found|共收集 0", collect_output):
        raise ConfigError(
            f"搜狗微信搜索未返回关于「{topic}」的有效文章。"
            f"请尝试: 1) 换一个更通用的关键词 2) 稍后重试 3) 检查网络连接"
        )

    # ── Phase 2a: generate initial report ───────────────────
    report_task = create_report_task(reporter, collect_task, analyze_task, fact_check_task)
    init_crew = Crew(
        agents=[reporter],
        tasks=[report_task],
        process=Process.sequential,
        verbose=True,
    )
    init_result = init_crew.kickoff(inputs=inputs)
    report = init_result.tasks_output[0].raw if len(init_result.tasks_output) > 0 else ""

    # ── Phase 2b: mandatory correction round ────────────────
    if "✅ 所有" in fc_output and "❌" not in fc_output:
        # fact-check passed clean — skip correction
        audited_report_task = report_task
    else:
        fix_task = Task(
            description=(
                "逐条修正以下事实核查发现的每一个错误，不允许跳过。\n"
                "修正后输出完整的最终报告。\n\n"
                "🚨 核心规则:\n"
                "- 报告必须严格遵循 5 部分结构（摘要/核心主题/风险/结论/参考来源）\n"
                "- 参考来源的 URL 必须从以下原始文章中逐条复制，绝对禁止使用 example.com\n"
                "- **不要在报告中显示任何修正清单或修正标注** — 这些是内部信息\n"
                "- 如果没有错误，直接输出 Phase 2a 生成的报告\n\n"
                "## 需要修正的错误\n"
                f"{fc_output}"
            ),
            expected_output="最终报告的 Markdown，严格 5 部分，无修正残留，URL 真实。",
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

        # ── Phase 2c: conditional second correction if ≥3 remain ──
        remaining = _count_errors(fc2)
        if remaining >= 3:
            fix_task2 = Task(
                description=(
                    "上一轮还有以下错误未完全修正，必须逐条彻底修正。\n"
                    "输出最终报告，严格 5 部分，不显示任何修正标注。\n\n"
                    f"{fc2}"
                ),
                expected_output="最终 Markdown 报告，5 部分，URL 真实，无修正残留。",
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

    report = _validate_report(report, topic)
    return report, audit
