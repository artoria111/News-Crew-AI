from crewai import Task


def create_tasks(collector, analyzer):
    collect_task = Task(
        description=(
            "Search for the latest news articles about the topic: {topic}.\n"
            "Use WeChatSearchTool once with the topic as the query.\n\n"
            "HARD CONSTRAINTS:\n"
            "- Select up to {max_articles} most relevant articles from the last {days_back} days.\n"
            "- **最少 3 个不同信源** — 单一信源占比 ≤ 50%\n"
            "- **不遗漏任何匹配文章**\n"
            "- **每条必须记录发布日期。** 若搜索工具返回了日期，直接使用。"
            "若未返回，标注\"日期未知\"。**绝对禁止编造或推断日期。**\n\n"
            "OUTPUT FORMAT:\n"
            "- 顶部: \"共收集 X 篇文章，来自 Y 个信源\" + 信源分布表\n"
            "- 每条必须包含: 编号, 标题, URL, 信源名称(微信公众号), 发布日期, 原文摘要, 1-2条要点\n"
            "- 若少于 3 个信源或某来源 > 50%，顶部标注 \"⚠️ 信源多样性不足\""
        ),
        expected_output=(
            "A numbered list of news articles. Each entry includes: number, title, "
            "URL, source name, publication date (actual date or '日期未知'), "
            "original snippet, and 1-2 key bullet points."
        ),
        agent=collector,
    )

    analyze_task = Task(
        description=(
            "Review ALL collected news articles and extract their core essence.\n\n"
            "## 核心原则\n"
            "**你的任务是提炼精华，不是堆砌原文。** 用最简洁的语言概括每篇文章的"
            "核心信息，聚焦于\"发生了什么、为什么重要、有什么影响\"。"
            "避免冗长的引用和主观评价，让读者快速抓住要点。\n\n"
            "## STEP 0 — 完整性校验 (MANDATORY FIRST STEP)\n"
            "逐条核对所有收集到的文章，输出: \"完整性校验: 共收到 X 篇文章，全部纳入分析\"\n\n"
            "## STEP 1 — 信源多样性评估\n"
            "统计信源分布。若 <3 个来源 或 单一来源 > 50%，输出:\n"
            "\"⚠️ 信源多样性不达标（需≥3来源且单一来源≤50%）。\"\n"
            "若达标，输出: \"✅ 信源多样性达标: X 个来源，最高占比 Y%。\"\n\n"
            "## STEP 2 — 精华提炼\n"
            "按以下维度组织，每项控制在 3-5 句话以内:\n"
            "1. **核心主题**: 跨文章归纳 2-3 个最突出的主题\n"
            "2. **关键事实**: 提取值得关注的数据、事件、时间节点\n"
            "3. **值得关注的趋势**: 从多篇文章中浮现的方向性信号\n"
            "4. **风险信号**: 文章中暗示的市场/技术/政策风险（如有）\n\n"
            "全部使用中文。简洁优于冗长。"
        ),
        expected_output=(
            "A concise Chinese analysis: 完整性校验, 信源评估, "
            "核心主题 (2-3 items), 关键事实 (bullet points), "
            "趋势信号, 风险信号. Each section 3-5 sentences max. "
            "Minimal quoting, maximum insight density."
        ),
        agent=analyzer,
        context=[collect_task],
    )

    return collect_task, analyze_task


def create_report_task(reporter, collect_task, analyze_task, fact_check_task):
    return Task(
        description=(
            "Using the fact-checked analysis, create a professional markdown report.\n\n"
            "## 核心原则\n"
            "**在撰写报告前，必须先阅读事实核查结果（Stage 3），** "
            "根据核查结论修正报告中所有被标记为错误的内容。\n"
            "**报告要有足够的信息量，让读者理解事件的来龙去脉。**\n"
            "每篇文章写 4-6 句话，说清楚：发生了什么 → 为什么重要 → 意味着什么。\n"
            "不堆砌原文引用，但细节要足够具体（具体公司名、具体数据、具体事件）。\n"
            "**绝对禁止编造任何信息。** 日期、数字、URL 必须来自原始材料，宁可不写也不编造。\n\n"
            "## 🚨 URL 防篡改规则 (CRITICAL)\n"
            "**每条参考来源的 URL 必须从 Stage 1 收集的原始文章中逐条复制。**\n"
            "绝对禁止使用 https://example.com 或任何占位符 URL。\n"
            "如果找不到某篇文章的 URL，在参考来源中标注\"URL缺失\"而非编造。\n\n"
            "## 信源多样性声明\n"
            "**仅当**分析中标注 \"⚠️ 信源多样性不达标\" 时，在报告顶部加声明。\n"
            "若标注 \"✅ 信源多样性达标\"，则不加。\n\n"
            "## 报告结构 (必须按顺序)\n"
            "1. **报告标题**: 包含主题\n"
            "2. **执行摘要**: 2-3段总结核心发现和关键趋势\n"
            "3. **信源概况**: 来源数量和多样性\n"
            "4. **核心主题**: 每个主题写清楚前因后果，4-6句话。包含具体细节\n"
            "5. **风险与挑战** (MANDATORY): 独立章节\n"
            "6. **关键结论**: 3-5条可操作的洞察\n"
            "7. **参考来源**: 编号列表，每条格式:\n"
            "   `1. [文章标题](从Stage1复制URL) — 来源: XX公众号, 日期: YYYY-MM-DD`\n"
            "8. **免责声明** (MANDATORY): 在报告最后加上:\n"
            "   `---`\n"
            "   `> ⚠️ **免责声明:** 本报告由 AI 自动生成，基于搜狗微信搜索的公开文章，仅供信息参考，不构成任何建议。报告中可能存在遗漏或偏差，请读者自行核实关键信息。`\n\n"
            "Markdown 格式，中文撰写。"
        ),
        expected_output=(
            "A complete Chinese markdown report with 8 sections, concise content, "
            "and reference list with real URLs copied from Stage 1. "
            "No fabricated information. Disclaimer only when diversity fails."
        ),
        agent=reporter,
        context=[collect_task, analyze_task, fact_check_task],
    )


def create_fact_check_task(fact_checker, collect_task, analyze_task):
    return Task(
        description=(
            "You are the fact-checking gatekeeper between analysis and report.\n\n"
            "## YOUR MISSION\n"
            "Cross-verify every key claim in the analysis against the original "
            "collected articles. Catch distortions, misinterpretations, and "
            "fabricated information.\n\n"
            "## CHECKLIST\n"
            "1. **URL真实性**: Are ALL URLs in references copied from Stage 1? "
            "Flag ANY https://example.com or placeholder URLs as ERRORS.\n"
            "2. **日期真实性**: Check for fabricated dates. OK if '日期未知'.\n"
            "3. **实体名称**: Company/person/product names correct?\n"
            "4. **数字数据**: Statistics and amounts accurate vs source?\n"
            "5. **事件描述**: Does analysis correctly represent what happened? "
            "e.g. \"invested in\" ≠ \"launched a division\"\n"
            "6. **遗漏文章**: All collected articles reflected in analysis? "
            "List missing ones by title.\n"
            "7. **编造检测**: Unsourced claims/dates/data? Flag ALL.\n\n"
            "## OUTPUT FORMAT\n"
            "### ✅ 验证通过\n"
            "List verified claims with source reference.\n\n"
            "### ❌ 发现错误 (if any)\n"
            "For each error:\n"
            "- 错误陈述: [what was said]\n"
            "- 原文实际: [what source actually says]\n"
            "- 修正建议: [corrected statement]\n\n"
            "### 📋 核查总结\n"
            "- 核查条目: X | 通过: X | 错误: X\n"
            "- 整体可信度: [高/中/低]\n\n"
            "No errors? State: \"✅ 所有事实核查通过。\""
        ),
        expected_output=(
            "A structured fact-check report: verified claims, flagged errors "
            "(with source evidence), summary, and credibility rating. In Chinese."
        ),
        agent=fact_checker,
        context=[collect_task, analyze_task],
    )


def create_audit_task(auditor, collect_task, analyze_task, fact_check_task, report_task):
    return Task(
        description=(
            "You are auditing a 4-stage news analysis pipeline:\n"
            "Stage 1: 新闻收集 → Stage 2: 精华提炼 → Stage 3: 事实核查 → Stage 4: 报告生成\n\n"
            "Evaluate across 6 dimensions (score 1-10):\n\n"
            "1. **新闻源可信度** (权重 ×1.5): ≥3 sources? Any source > 50%?\n"
            "2. **新闻质量** (权重 ×1.5): relevance, information density. "
            "**Note: 不因日期缺失扣分，这不是 Agent 的错。**\n"
            "3. **信息提炼与事实核查** (权重 ×1.0): Was article essence accurately captured? "
            "Did fact-checker catch errors? Concise > verbose.\n"
            "4. **分析精准度** (权重 ×0.5): Are themes correctly identified? "
            "Risk signals spotted? **Subjective analysis is secondary — "
            "accurate information extraction matters more.**\n"
            "5. **报告质量** (权重 ×1.5): Formatting, logical flow, "
            "diversity and AI disclaimers correct? "
            "Any example.com URLs or fabricated content?\n"
            "6. **数据流完整性** (权重 ×1.5): 逐篇追踪，制作文章对照表:\n"
            "   | Stage1编号 | 文章标题 | Stage2分析 | Stage3核查 | Stage4报告 | 备注 |\n"
            "   |-----------|---------|:---:|:---:|:---:|------|\n"
            "   | 1 | [标题] | ✅/❌ | ✅/❌ | ✅/❌ | |\n"
            "   ...\n"
            "   列出所有在传递中丢失或扭曲的文章标题。\n"
            "   **Verify all URLs are real (not example.com).**\n\n"
            "## 评分注意事项\n"
            "- **日期缺失不扣分** — \"日期未知\"是搜狗微信的限制，非 Agent 问题\n"
            "- **分析偏短不是问题** — 简洁 > 冗长\n"
            "- **报告正文有效内容越多越好** — 引用越少越好\n\n"
            "Output scorecard in Chinese Markdown:\n"
            "- **加权总分** (weighted average of 6 dimensions)\n"
            "- **逐项评分表**: score (1-10) × weight + comment\n"
            "- **三大亮点**\n"
            "- **三项改进建议**"
        ),
        expected_output=(
            "A structured Chinese audit scorecard: weighted overall score, "
            "per-dimension scores (with weights) and comments, "
            "top 3 strengths, top 3 suggestions."
        ),
        agent=auditor,
        context=[collect_task, analyze_task, fact_check_task, report_task],
    )
