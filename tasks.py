from crewai import Task


def create_tasks(collector, analyzer, reporter):
    collect_task = Task(
        description=(
            "Search for the latest news articles about the topic: {topic}.\n"
            "Use RSSNewsSearchTool once with the topic as the query.\n\n"
            "CRITICAL RULES:\n"
            "- Select up to {max_articles} most relevant articles from the last {days_back} days.\n"
            "- **Must include articles from at least 2-3 different sources** (e.g. 36kr, 人民网, "
            "IT之家, China Daily, Yonhap, France24). NEVER use only one source.\n"
            "- **Do NOT drop any matching article** — include every relevant result. If the RSS "
            "tool returned fewer than {max_articles}, include all of them and note the count.\n"
            "- For each article: record title, URL, source name, publication date, "
            "the RSS-provided summary, and 1-2 key insight bullet points.\n"
            "- Use the RSS summary as-is — do NOT scrape full articles.\n"
            "- At the top of your output, state: \"共收集 X 篇文章，来自 Y 个信源\"\n"
            "- If fewer than 2 distinct sources are available, state this explicitly as a warning."
        ),
        expected_output=(
            "A structured list of news articles. Each entry includes: number, title, "
            "source URL, publication date, source name, RSS summary, and 1-2 key bullet "
            "points. Header states total count and source count."
        ),
        agent=collector,
    )

    analyze_task = Task(
        description=(
            "Review ALL collected news articles and perform a thorough analysis.\n\n"
            "## STEP 0 — 完整性校验 (MANDATORY FIRST STEP)\n"
            "Go through EVERY article in the collection one by one. Verify that each "
            "article's key information is preserved. If any article is missing, note it "
            "explicitly. At the start of your analysis, output:\n"
            "\"完整性校验: 共收到 X 篇文章，全部纳入分析 / 缺失 Y 篇: [列出缺失文章标题]\"\n\n"
            "## STEP 1 — 信源多样性评估\n"
            "Count distinct sources. If only 1 source exists, output this exact warning at "
            "the TOP of your analysis:\n"
            "\"⚠️ 警告: 本报告基于单一信源，结论可能存在偏差，建议参考更多来源。\"\n\n"
            "## STEP 2 — 深度分析 (must cover ALL of the following)\n"
            "1. **主题与趋势**: Identify cross-cutting themes; validate each theme against 2+ articles.\n"
            "2. **矛盾与争议**: Find conflicting viewpoints or disagreements between sources. "
            "If none exist, explain why (e.g. single source, consensus topic).\n"
            "3. **风险与挑战** (MANDATORY): For every major trend or claim, analyze potential "
            "risks — market acceptance, technology maturity, regulatory uncertainty, "
            "cost barriers, competitive threats, etc. If an article mentions pricing "
            "(e.g. subscription fees), analyze market acceptance risk. If it mentions "
            "new technology, assess technical readiness and adoption barriers.\n"
            "4. **关键数据与事实**: Extract specific numbers, statistics, dates.\n"
            "5. **情绪评估**: Overall sentiment (positive/negative/neutral) across sources.\n"
            "6. **专家观点**: Notable quotes or authoritative opinions.\n\n"
            "All analysis in Chinese."
        ),
        expected_output=(
            "A structured Chinese analysis document with mandatory sections:\n"
            "- 完整性校验 (article count verification)\n"
            "- 信源多样性评估 (source diversity, with single-source warning if applicable)\n"
            "- 主题与趋势 (validated against multiple articles)\n"
            "- 矛盾与争议 (conflicting viewpoints)\n"
            "- 风险与挑战 (risk analysis for every major trend — MANDATORY)\n"
            "- 关键数据与事实\n"
            "- 情绪评估\n"
            "- 专家观点"
        ),
        agent=analyzer,
        context=[collect_task],
    )

    report_task = Task(
        description=(
            "Using the analysis provided, create a professional markdown report.\n\n"
            "CRITICAL: If the analysis contains a single-source warning, you MUST "
            "include this disclaimer at the VERY TOP of the report (before the title):\n"
            "\"⚠️ **免责声明:** 本报告基于单一信源，结论可能存在偏差。建议读者参考更多信息渠道。\"\n\n"
            "The report MUST include these sections IN ORDER:\n"
            "1. **报告标题**: Clear, engaging title with topic and date\n"
            "2. **执行摘要**: 2-3 paragraphs summarizing key findings AND key risks\n"
            "3. **信源概况**: Brief note on source range, count, and quality; single-source warning if applicable\n"
            "4. **核心主题**: Detailed exploration of each major theme with supporting evidence\n"
            "5. **风险与挑战** (MANDATORY): Dedicated section analyzing risks, challenges, "
            "and counterpoints for each major finding. Include market risks, technology "
            "risks, and adoption barriers where relevant.\n"
            "6. **关键结论**: 3-5 actionable takeaways\n"
            "7. **参考来源**: Numbered list of ALL referenced articles with URLs\n\n"
            "Format cleanly in markdown. Use headings, bullet points, and bold. "
            "Write the entire report in Chinese."
        ),
        expected_output=(
            "A complete Chinese markdown report with all 7 mandatory sections, "
            "proper formatting, source attribution, and single-source disclaimer "
            "if applicable."
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
            "1. **新闻源可信度** — source credibility and diversity. Is the "
            "mandatory 2-3 source rule followed?\n"
            "2. **新闻质量** — article relevance, timeliness, information density\n"
            "3. **摘要质量** — accuracy and completeness; are ALL collected articles "
            "included in the analysis? (Check 完整性校验)\n"
            "4. **分析质量** — depth of themes, contradiction detection, is the "
            "mandatory 风险与挑战 section present and substantive?\n"
            "5. **报告质量** — formatting, logical flow, readability, single-source "
            "disclaimer present if needed\n"
            "6. **数据流完整性** — Verify: Are ALL articles from Stage 1 present in "
            "Stage 2 analysis? Any information lost or distorted between stages? "
            "List any missing articles by title.\n"
            "7. **效率评估** — pipeline efficiency based on data volume and processing\n\n"
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
