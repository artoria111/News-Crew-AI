import io
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

from crewai import Crew, Process, Task
from dotenv import load_dotenv

from agents import (
    create_analyzer,
    create_auditor,
    create_collector,
    create_fact_checker,
    create_llm,
    create_reporter,
)
from tasks import (
    create_audit_task,
    create_fact_check_task,
    create_report_task,
    create_tasks,
)
from tools import BochaSearchTool

# UI display helpers — built from the tools/llm that run_crew() actually uses.
TOOL_CLASS = BochaSearchTool
SEARCH_PROVIDER = "博查全网搜索"


def _llm_provider_label() -> str:
    base = os.getenv("LLM_BASE_URL", "").lower()
    model = os.getenv("LLM_MODEL_NAME", "LLM")
    vendors = [
        ("deepseek.com", "DeepSeek"),
        ("minimaxi.com", "MiniMax"),
        ("siliconflow.cn", "SiliconFlow"),
        ("openai.com", "OpenAI"),
        ("bigmodel.cn", "智谱 GLM"),
        ("dashscope.aliyuncs.com", "通义千问"),
    ]
    vendor = next((v for k, v in vendors if k in base), model)
    return f"{vendor} ({model})"


LLM_PROVIDER = _llm_provider_label()


class ConfigError(Exception):
    pass


def validate_env():
    missing = [v for v in ["LLM_API_KEY"] if not os.getenv(v)]
    if missing:
        raise ConfigError("缺少 LLM_API_KEY，请在 .env 文件中配置。")


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


class _Tee:
    """A file-like wrapper that writes to both a real stream and a buffer.

    Used by redirect_stdout/redirect_stderr so the CrewAI verbose output
    still appears in the terminal while ALSO being captured to a log file.
    ANSI escape codes are stripped from the buffered copy only — the terminal
    sees the original (colored) text.
    """
    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, real, buf):
        self._real = real
        self._buf = buf

    def write(self, s):
        self._real.write(s)
        self._buf.write(self._ANSI_RE.sub("", s))
        return len(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self):
        return False


# ── per-agent token tracking ───────────────────────────────
# CrewAI's LLMCallCompletedEvent fires once per LLM call with `usage` (token
# counts) and `agent_role` / `task_name`. We aggregate per agent and per
# task so the Streamlit UI can show "who spent the tokens".
from crewai.events import BaseEventListener
from crewai.events.event_types import LLMCallCompletedEvent, LLMCallFailedEvent


class TokenUsageListener(BaseEventListener):
    def __init__(self):
        super().__init__()
        # {agent_role: {"prompt": int, "completion": int, "calls": int, "tasks": set}}
        self.by_agent: dict = {}
        # {task_name: {"prompt": int, "completion": int, "calls": int, "agent": str}}
        self.by_task: dict = {}
        self.failed_calls: int = 0

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _on_completed(source, event: LLMCallCompletedEvent):
            usage = event.usage or {}
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            cached = int(usage.get("cached_prompt_tokens") or 0)
            agent_role = event.agent_role or "unknown"
            # some LLMs (M3) leak task description into event.task_name.
            # collapse anything that looks like prompt echo into a short slug.
            raw_task = event.task_name or "unknown"
            if len(raw_task) > 60 or any(
                bad in raw_task
                for bad in ("🚨", "核心规则", "禁止使用", "修正清单", "Phase 2a")
            ):
                task_name = f"<{agent_role}>"
            else:
                task_name = raw_task

            a = self.by_agent.setdefault(
                agent_role, {"prompt": 0, "completion": 0, "cached": 0, "calls": 0, "tasks": set()}
            )
            a["prompt"] += prompt
            a["completion"] += completion
            a["cached"] += cached
            a["calls"] += 1
            a["tasks"].add(task_name)

            t = self.by_task.setdefault(
                task_name, {"prompt": 0, "completion": 0, "cached": 0, "calls": 0, "agent": agent_role}
            )
            t["prompt"] += prompt
            t["completion"] += completion
            t["cached"] += cached
            t["calls"] += 1

        @crewai_event_bus.on(LLMCallFailedEvent)
        def _on_failed(source, event: LLMCallFailedEvent):
            self.failed_calls += 1


def _format_token_table(by_agent: dict, by_task: dict, failed: int) -> str:
    """Render a markdown table of token usage per agent and per task."""
    lines = ["## 📊 Token 用量统计\n"]

    lines.append("### 按 Agent 统计\n")
    lines.append("| Agent | 调用次数 | Prompt | Cached | Completion | 关联任务 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for role, m in sorted(by_agent.items(), key=lambda x: -(x[1]["prompt"] + x[1]["completion"])):
        tasks = ", ".join(sorted(m["tasks"])) or "-"
        lines.append(f"| {role} | {m['calls']} | {m['prompt']} | {m['cached']} | {m['completion']} | {tasks} |")

    lines.append("\n### 按 Task 统计\n")
    lines.append("| Task | Agent | 调用 | Prompt | Cached | Completion |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for task, m in sorted(by_task.items(), key=lambda x: -(x[1]["prompt"] + x[1]["completion"])):
        lines.append(f"| {task} | {m['agent']} | {m['calls']} | {m['prompt']} | {m['cached']} | {m['completion']} |")

    total_prompt = sum(m["prompt"] for m in by_agent.values())
    total_cached = sum(m["cached"] for m in by_agent.values())
    total_completion = sum(m["completion"] for m in by_agent.values())
    lines.append(
        f"\n**总计**: Prompt {total_prompt} (其中 Cached {total_cached})"
        f" + Completion {total_completion} = **{total_prompt + total_completion} token**"
        f" | **失败调用**: {failed}\n"
    )
    return "\n".join(lines)


def _count_errors(fact_check_output: str) -> int:
    """Parse fact-check output to count discovered errors."""
    errors = re.findall(r"错误陈述", fact_check_output)
    cross_marks = fact_check_output.count("❌")
    return max(len(errors), cross_marks)


# prompt fragments that, if echoed back into the report, indicate the LLM
# is leaking its task description instead of producing the clean report.
# Used to scrub accidental prompt echoes from the final user-facing output.
_PROMPT_ECHO_MARKERS = (
    "报告必须严格",
    "禁止使用 example.com",
    "不要在报告中显示任何修正清单",
    "Markdown 格式，中文撰写",
    "🚨 核心规则",
    "🚨 URL 规则",
    "严格遵循 5 部分",
)


def _strip_prompt_echo(text: str) -> str:
    """Remove lines / blocks that look like a leaked task description.

    Two passes:
    1. Drop leading echo: if the report starts with description fragments
       ("### 1. 执行摘要" → "### 2. 核心主题" → ...) before the actual
       "# <report title>", discard everything up to and including the
       last "### 5. 参考来源" / "报告末尾固定加上" line.
    2. Drop inline echoes: any line matching a known prompt marker
       starts a drop-window until a real-content marker appears.
    """
    lines = text.splitlines()

    # ── pass 1: leading-echo detection ──────────────────────
    # the legitimate report's H1 starts with "# " (single hash) and is
    # followed by a real title. Everything before that, AND everything
    # between the H1 and the next blank-line-followed-by-body, is suspect
    # if it contains the description's 5-section template.
    section_titles = (
        "### 1. 执行摘要",
        "### 2. 核心主题",
        "### 3. 风险与挑战",
        "### 4. 关键结论",
        "### 5. 参考来源",
        "报告末尾固定加上",
        "报告结构（固定",
        "报告结构(固定",
    )
    h1_idx = None
    for idx, ln in enumerate(lines):
        if ln.startswith("# ") and not ln.startswith("## "):
            # skip a stray # appearing inside echo
            if h1_idx is None:
                h1_idx = idx
                break

    if h1_idx is not None and h1_idx > 0:
        head = "\n".join(lines[:h1_idx])
        # if head contains any section template title, discard all of it
        if any(sec in head for sec in section_titles):
            lines = lines[h1_idx:]

    # ── pass 2: inline-echo detection ──────────────────────
    out_lines = []
    drop_block = False
    for ln in lines:
        stripped = ln.strip()
        # start dropping a block when we hit a known prompt instruction
        if any(marker in stripped for marker in _PROMPT_ECHO_MARKERS):
            drop_block = True
            continue
        # stop dropping once we reach a clearly-user-facing line
        if drop_block and (
            stripped.startswith(("#", ">", "---", "|", "1.", "2.", "3.", "4.", "5.", "["))
            or "http" in stripped
            or len(stripped) > 80
        ):
            drop_block = False
        if drop_block:
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip() + "\n"


def _validate_report(report: str, query: str) -> str:
    """Post-process: fix placeholder URLs and append real reference list if needed."""
    has_placeholder = bool(
        re.search(r"example\.com|example\.org|placeholder", report, re.IGNORECASE)
    )
    if not has_placeholder:
        return report

    # Re-run search to get real URLs
    try:
        tool = TOOL_CLASS()
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


def run_crew(topic: str, max_articles: int = 5, days_back: int = 7):
    load_dotenv()
    validate_env()

    # capture all stdout/stderr (CrewAI verbose output, llm logs, etc.)
    # to a per-run log file so every test run is inspectable after the fact.
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^\w\-]+", "_", topic)[:30] or "run"
    log_path = os.path.join(LOG_DIR, f"crew_{safe_topic}_{stamp}.log")
    log_buffer = io.StringIO()
    log_file = open(log_path, "w", encoding="utf-8")
    # tee stdout/stderr: writes go to both the real terminal (so users see
    # live progress) and the in-memory buffer (flushed to log_file on exit).
    stdout_tee = _Tee(sys.stdout, log_buffer)
    stderr_tee = _Tee(sys.stderr, log_buffer)

    # attach a per-run token usage listener (auto-registers on bus)
    token_listener = TokenUsageListener()

    llm = create_llm()
    search_tool = TOOL_CLASS()
    inputs = {
        "topic": topic,
        "max_articles": str(max_articles),
        "days_back": str(days_back),
    }

    collector = create_collector(search_tool, llm)
    analyzer = create_analyzer(llm)
    reporter = create_reporter(llm)

    # ── pre-search: some LLMs (e.g. MiniMax M3) don't serialize tool args
    # correctly, so the agent invents fake results. Run the search directly
    # here and inject the raw output into the task description.
    print("🔍 直接预搜索 (绕过 agent tool-calling)...")
    raw_search = search_tool._run(topic)
    inputs["raw_articles"] = raw_search

    collect_task, analyze_task = create_tasks(collector, analyzer)
    fact_checker = create_fact_checker(llm)
    auditor = create_auditor(llm)

    try:
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            report, audit = _run_crew_pipeline(
                topic=topic,
                max_articles=max_articles,
                days_back=days_back,
                inputs=inputs,
                collector=collector,
                analyzer=analyzer,
                reporter=reporter,
                collect_task=collect_task,
                analyze_task=analyze_task,
                fact_checker=fact_checker,
                auditor=auditor,
            )
    finally:
        log_file.write(log_buffer.getvalue())
        log_file.close()
        print(f"\n📝 运行日志已保存: {log_path}")

    # append token usage summary to the report
    token_report = _format_token_table(
        token_listener.by_agent, token_listener.by_task, token_listener.failed_calls
    )
    report = report + "\n\n---\n\n" + token_report
    return report, audit


def _run_crew_pipeline(
    topic,
    max_articles,
    days_back,
    inputs,
    collector,
    analyzer,
    reporter,
    collect_task,
    analyze_task,
    fact_checker,
    auditor,
):

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
    collect_output = (
        result1.tasks_output[0].raw if len(result1.tasks_output) > 0 else ""
    )
    if (
        re.search(
            r"共收集\s*0\s*篇|No (WeChat|web) articles found|共收集 0",
            collect_output,
        )
        or "未找到相关" in collect_output
    ):
        raise ConfigError(
            f"{SEARCH_PROVIDER}未返回关于「{topic}」的有效文章。"
            f"请尝试: 1) 换一个更通用的关键词 2) 稍后重试 3) 检查网络连接或 API Key"
        )

    # ── Phase 2a: generate initial report ───────────────────
    report_task = create_report_task(
        reporter, collect_task, analyze_task, fact_check_task
    )
    init_crew = Crew(
        agents=[reporter],
        tasks=[report_task],
        process=Process.sequential,
        verbose=True,
    )
    init_result = init_crew.kickoff(inputs=inputs)
    report = (
        init_result.tasks_output[0].raw if len(init_result.tasks_output) > 0 else ""
    )

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
            report = (
                result2b.tasks_output[0].raw
                if len(result2b.tasks_output) > 0
                else report
            )
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

    report = _strip_prompt_echo(report)
    report = _validate_report(report, topic)
    return report, audit
