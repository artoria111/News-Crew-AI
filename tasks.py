from crewai import Task


def create_tasks(collector, analyzer, reporter):
    collect_task = Task(
        description=(
            "Search for the latest news articles about the topic: {topic}.\n"
            "Use RSSNewsSearchTool once with the topic as the query.\n"
            "From the results, select up to {max_articles} most relevant "
            "articles from the last {days_back} days.\n"
            "For each selected article:\n"
            "- Record the title, URL, source, and publication date\n"
            "- Use the summary already provided by RSS — do NOT scrape full articles\n"
            "- Add 1-2 bullet points of the most important insight from the summary\n\n"
            "Compile a clean, structured list. If no articles are found, "
            "state that clearly and suggest trying broader keywords."
        ),
        expected_output=(
            "A structured list of up to {max_articles} news articles. Each entry "
            "includes: title, source URL, publication date, source name, "
            "a brief Chinese summary, and 1-2 key bullet points."
        ),
        agent=collector,
    )

    analyze_task = Task(
        description=(
            "Review all collected news articles and perform a thorough analysis:\n"
            "1. Identify the main themes and topics emerging across multiple articles\n"
            "2. Note any conflicting viewpoints or disagreements between sources\n"
            "3. Highlight key facts, statistics, and data points\n"
            "4. Assess the overall sentiment (positive, negative, neutral) across sources\n"
            "5. Identify any notable quotes or expert opinions\n"
            "6. Evaluate source diversity and potential biases\n\n"
            "Provide balanced, well-organized analysis in Chinese. "
            "If the collected article list is empty, state that no articles "
            "were found and suggest trying different search terms."
        ),
        expected_output=(
            "A structured analysis document with sections:\n"
            "- Key themes and trends (with cross-source validation)\n"
            "- Conflicting viewpoints (if any)\n"
            "- Key facts and statistics\n"
            "- Sentiment analysis\n"
            "- Notable quotes and expert opinions\n"
            "- Source diversity assessment\n\n"
            "All written in Chinese."
        ),
        agent=analyzer,
        context=[collect_task],
    )

    report_task = Task(
        description=(
            "Using the analysis provided, create a professional markdown report.\n"
            "The report must include:\n"
            "1. **Title**: A clear, engaging title including the topic and date\n"
            "2. **Executive Summary**: 2-3 paragraphs summarizing key findings\n"
            "3. **Key Themes**: Detailed exploration of each major theme with evidence\n"
            "4. **Source Analysis**: Brief note on the range and quality of sources\n"
            "5. **Key Takeaways**: 3-5 actionable or important takeaways\n"
            "6. **Sources**: A numbered list of all referenced articles with URLs\n\n"
            "Format cleanly in markdown. Use headings, bullet points, and bold "
            "text for emphasis where appropriate. Write the entire report in Chinese."
        ),
        expected_output=(
            "A complete markdown document ready for publication. Well-formatted "
            "with clear section headings, proper source attribution, and "
            "professional language. Written in Chinese."
        ),
        agent=reporter,
        context=[analyze_task],
    )

    return collect_task, analyze_task, report_task


def create_audit_task(auditor, collect_task, analyze_task, report_task):
    return Task(
        description=(
            "You are auditing a 3-stage news analysis pipeline. Review the "
            "outputs from each stage and evaluate them systematically.\n\n"
            "Evaluate across these 7 dimensions, scoring each 1-10:\n\n"
            "1. **新闻源可信度** (Source Credibility) — Are the RSS sources "
            "authoritative and diverse? Are outlets like 36kr and ThePaper "
            "legitimate news sources?\n"
            "2. **新闻质量** (Article Quality) — Are the collected articles "
            "relevant to the search topic, timely, and information-dense?\n"
            "3. **摘要质量** (Summary Quality) — Did the collector accurately "
            "capture key points without distortion or omission?\n"
            "4. **分析质量** (Analysis Quality) — Are cross-cutting themes "
            "correctly identified? Are contradictions surfaced? Is sentiment "
            "assessment reasonable?\n"
            "5. **报告质量** (Report Quality) — Is the markdown well-formatted? "
            "Is the logical flow clear? Is the language professional?\n"
            "6. **效率评估** (Efficiency) — Based on the volume of collected "
            "data and processing speed, how efficient was the pipeline?\n"
            "7. **数据流完整性** (Data Flow Integrity) — Was information "
            "preserved or lost between stages? Any distortion detected?\n\n"
            "Output a complete quality audit scorecard in Chinese Markdown:\n"
            "- **总体评分** (average of 7 dimensions)\n"
            "- **逐项评分表**: each dimension with score (1-10) + 1-2 sentence comment\n"
            "- **三大亮点**: Top 3 strengths observed\n"
            "- **三项改进建议**: Top 3 actionable improvement suggestions"
        ),
        expected_output=(
            "A structured quality audit scorecard in Chinese Markdown, "
            "containing: overall score, per-dimension scores with comments, "
            "top 3 strengths, and top 3 improvement suggestions."
        ),
        agent=auditor,
        context=[collect_task, analyze_task, report_task],
    )
