from crewai import Task


def create_tasks(collector, analyzer, reporter):
    collect_task = Task(
        description=(
            "Search for the latest news articles about the topic: {topic}.\n"
            "Use WeChatSearchTool once with the topic as the query.\n\n"
            "HARD CONSTRAINTS:\n"
            "- Select up to {max_articles} most relevant articles from the last {days_back} days.\n"
            "- **最少 3 个不同信源** — 单一信源占比 ≤ 50%\n"
            "- **不遗漏任何匹配文章**\n"
            "- **每条必须记录发布日期（YYYY-MM-DD）** — 这是必填字段，不可省略。\n"
            "  若搜索工具未返回日期，从 URL、文章内容上下文或标题推断并标注\"推断日期\"。\n"
            "  若完全无法推断，标注\"日期未知（无法推断）\"并说明原因。\n\n"
            "OUTPUT FORMAT:\n"
            "- 顶部: \"共收集 X 篇文章，来自 Y 个信源\" + 信源分布表\n"
            "- 每条必须包含: 编号, 标题, URL, 信源名称(微信公众号), **发布日期**, 原文摘要, 1-2条要点\n"
            "- 若少于 3 个信源或某来源 > 50%，顶部标注 \"⚠️ 信源多样性不足\""
        ),
        expected_output=(
            "A numbered list of news articles. Each entry MUST include: number, title, "
            "URL, source name, publication date (YYYY-MM-DD or '日期未知'), "
            "original RSS snippet, and 1-2 key bullet points."
        ),
        agent=collector,
    )

    analyze_task = Task(
        description=(
            "Review ALL collected news articles and perform a thorough analysis.\n\n"
            "## STEP 0 — 完整性校验 (MANDATORY FIRST STEP)\n"
            "逐条核对所有收集到的文章，输出: \"完整性校验: 共收到 X 篇文章，全部纳入分析\"\n\n"
            "## STEP 1 — 信源多样性评估\n"
            "统计信源分布。若 <3 个来源 或 单一来源 > 50%，输出:\n"
            "\"⚠️ 信源多样性不达标（需≥3来源且单一来源≤50%）。\"\n"
            "若达标，输出: \"✅ 信源多样性达标: X 个来源，最高占比 Y%。\"\n\n"
            "## STEP 2 — 深度分析\n"
            "**关键规则: 每条分析结论必须直接引用原文来支撑。**\n"
            "不要仅依赖 Stage 1 的\"要点\"提炼 — 要点可能已有偏差。\n"
            "引用格式: 「原文: \"...[摘录原文关键句]...\"」(来源: XX公众号)\n\n"
            "必须覆盖以下所有维度:\n"
            "1. **主题与趋势**: 每个主题引用 2+ 篇文章原文作为证据\n"
            "2. **矛盾与争议**: 发现不同信源间的观点冲突，引用双方原文\n"
            "3. **风险与挑战** (MANDATORY): 针对每条主要趋势分析风险 — "
            "市场接受度、技术成熟度、监管不确定性、成本壁垒、竞争威胁等\n"
            "4. **关键数据与事实**: 提取具体数字，标注来源\n"
            "5. **情绪评估**: 总体情绪倾向\n"
            "6. **专家观点**: 值得关注的引用或权威意见\n\n"
            "全部使用中文输出。"
        ),
        expected_output=(
            "A structured Chinese analysis with mandatory sections: "
            "完整性校验, 信源多样性评估, 主题与趋势(with原文引用), "
            "矛盾与争议, 风险与挑战, 关键数据与事实, 情绪评估, 专家观点. "
            "Every analytical claim is backed by a direct quote from the original article."
        ),
        agent=analyzer,
        context=[collect_task],
    )

    report_task = Task(
        description=(
            "Using the fact-checked analysis, create a professional markdown report.\n\n"
            "## 核心原则 — 严格基于证据\n"
            "**绝对禁止编造或引入任何未经原始材料证实的信息。**\n"
            "包括但不限于: 具体年份(如\"2026年\")、数字、人名、公司名、事件细节。\n"
            "报告中的每一个断言都必须能追溯到 Stage 1 的原始文章或 Stage 3 的事实核查结论。\n"
            "如果原始材料没有提供某个信息，宁可写\"未提及\"也绝不能编造。\n\n"
            "## 免责声明规则\n"
            "**仅当**分析中明确标注 \"⚠️ 信源多样性不达标\" 时，才在报告最顶部加免责声明。\n"
            "**若标注 \"✅ 信源多样性达标\"，则绝对不要加免责声明。**\n\n"
            "## 文章清单对照表 (MANDATORY — 防丢失机制)\n"
            "在报告末尾的\"参考来源\"之前，插入一个对照表:\n"
            "| Stage 1 编号 | 文章标题 | 是否在报告中引用 | 备注 |\n"
            "|-------------|---------|:---:|------|\n"
            "| 1 | [标题] | ✅ | |\n"
            "| 2 | [标题] | ✅ | |\n"
            "...\n"
            "逐篇核对 Stage 1 的每篇文章是否在报告中出现。若有遗漏，必须在备注中说明原因。\n\n"
            "## 报告结构 (必须按顺序)\n"
            "1. **报告标题**: 包含主题和日期\n"
            "2. **执行摘要**: 2-3段总结核心发现和主要风险\n"
            "3. **信源概况**: 来源数量和多样性评估\n"
            "4. **核心主题**: 每个主题附证据（引用原文）\n"
            "5. **风险与挑战** (MANDATORY): 独立章节\n"
            "6. **关键结论**: 3-5条可操作的洞察\n"
            "7. **文章清单对照表** (MANDATORY): 见上\n"
            "8. **参考来源**: 编号列表，含 URL\n\n"
            "Markdown 格式，中文撰写。"
        ),
        expected_output=(
            "A complete Chinese markdown report with 8 sections (including mandatory "
            "article checklist cross-reference table), proper formatting, source "
            "attribution, and ZERO fabricated claims. Disclaimer ONLY present when "
            "triggered by diversity failure."
        ),
        agent=reporter,
        context=[analyze_task],
    )

    return collect_task, analyze_task, report_task


def create_fact_check_task(fact_checker, collect_task, analyze_task):
    return Task(
        description=(
            "You are the fact-checking gatekeeper between analysis and report.\n\n"
            "## YOUR MISSION\n"
            "Cross-verify every key claim in the analysis against the original "
            "collected articles. This is the LAST LINE OF DEFENSE before the "
            "report goes out — errors that pass you become published mistakes.\n\n"
            "## CHECKLIST (verify ALL)\n"
            "1. **发布日期** (MANDATORY FIRST CHECK): Does every article have a date? "
            "Verify dates are accurate vs source. Flag any missing or incorrect dates.\n"
            "2. **实体名称**: Are company names, person names, product names correct?\n"
            "3. **数字数据**: Are all statistics, dates, amounts accurate vs source?\n"
            "4. **事件描述**: Does the analysis correctly represent what happened?\n"
            "   — e.g. If source says \"X company invested in Y\", analysis should "
            "NOT say \"X company launched Y division\"\n"
            "5. **引用准确性**: Are direct quotes faithful to the original?\n"
            "6. **遗漏文章**: Are ALL collected articles reflected in the analysis? "
            "List any missing by title.\n"
            "7. **编造检测**: Are there any claims, dates, or data in the analysis "
            "that cannot be traced back to the original articles? Flag ALL unsourced claims.\n\n"
            "## OUTPUT FORMAT\n"
            "### ✅ 验证通过\n"
            "List each verified claim with source reference.\n\n"
            "### ❌ 发现错误 (if any)\n"
            "For each error found:\n"
            "- 错误陈述: [what the analysis said]\n"
            "- 原文实际: [what the source actually said, with direct quote]\n"
            "- 修正建议: [corrected statement]\n\n"
            "### 📋 核查总结\n"
            "- 核查条目总数: X\n"
            "- 通过: X | 错误: X\n"
            "- 整体可信度: [高/中/低]\n\n"
            "If no errors found, clearly state: \"✅ 所有关键事实核查通过，分析结论与原文一致。\""
        ),
        expected_output=(
            "A structured fact-check report with verified claims, flagged errors "
            "(with source quotes), and an overall credibility assessment. In Chinese."
        ),
        agent=fact_checker,
        context=[collect_task, analyze_task],
    )


def create_audit_task(auditor, collect_task, analyze_task, fact_check_task, report_task):
    return Task(
        description=(
            "You are auditing a 4-stage news analysis pipeline:\n"
            "Stage 1: 新闻收集 → Stage 2: 深度分析 → Stage 3: 事实核查 → Stage 4: 报告生成\n\n"
            "Review the outputs from each stage and evaluate across 7 dimensions (1-10):\n\n"
            "1. **新闻源可信度** — ≥3 sources? Any source > 50%? Does EVERY article "
            "have a recorded date (or marked \"推断日期\")?\n"
            "2. **新闻质量** — relevance, timeliness, information density\n"
            "3. **摘要质量** — all articles included? Key info preserved?\n"
            "4. **分析质量** — themes, contradictions, risk analysis present? "
            "Direct quotes used as evidence?\n"
            "5. **事实核查质量** — Were errors caught? How many claims verified? "
            "Did the fact-checker actually cross-reference source material?\n"
            "6. **报告质量** — formatting, logical flow, disclaimer correctly applied? "
            "Is the mandatory article checklist table present? "
            "Any fabricated claims (dates, numbers, facts) not traceable to source?\n"
            "7. **数据流完整性** — Article-by-article trace: Stage1→Stage2→Stage3→Stage4. "
            "Any information lost or distorted? List missing articles by title.\n\n"
            "Output scorecard in Chinese Markdown:\n"
            "- **总体评分** (average of 7 dimensions)\n"
            "- **逐项评分表**: score (1-10) + comment\n"
            "- **三大亮点**\n"
            "- **三项改进建议**"
        ),
        expected_output=(
            "A structured quality audit scorecard in Chinese Markdown: "
            "overall score, per-dimension scores with comments, "
            "top 3 strengths, top 3 improvement suggestions."
        ),
        agent=auditor,
        context=[collect_task, analyze_task, fact_check_task, report_task],
    )
