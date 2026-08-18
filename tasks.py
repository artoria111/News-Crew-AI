from crewai import Task


def create_tasks(collector, analyzer):
    collect_task = Task(
        description=(
            "Format and filter the raw search results below for topic: {topic}.\n\n"
            "## RAW SEARCH RESULTS (from BochaSearchTool, already executed)\n\n"
            "{raw_articles}\n\n"
            "## YOUR JOB\n"
            "You are NOT calling any tool. The results are pre-supplied above.\n"
            "Just **select and reformat** them into the output format.\n\n"
            "HARD CONSTRAINTS:\n"
            "- Select up to {max_articles} most relevant articles from the last {days_back} days.\n"
            "- **最少 3 个不同信源** — 单一信源占比 ≤ 50%\n"
            "- **每条必须记录发布日期** — 直接复制搜索结果中的日期字段;"
            "若返回\"未知\"则标注\"日期未知\"。**绝对禁止编造或推断日期。**\n"
            "- **绝对禁止编造任何 URL 或标题** — 必须逐条从上面 RAW 数据复制。\n\n"
            "OUTPUT FORMAT:\n"
            "- 顶部: \"共收集 X 篇文章，来自 Y 个信源\" + 信源分布表\n"
            "- 每条必须包含: 编号, 标题, URL, 信源名称(网站名), 发布日期, 原文摘要, 1-2条要点\n"
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
            "**在撰写报告前，必须先阅读事实核查结果，** "
            "根据核查结论修正所有被标记为错误的内容。\n"
            "**绝对禁止编造任何信息。** 日期、数字、URL 必须来自原始材料。\n\n"
            "## 🚨 URL 规则\n"
            "**参考来源的 URL 必须从 Stage 1 原始文章逐条复制，禁止任何占位符。**\n"
            "报告必须严格 5 部分，不要添加任何额外章节、修正标注或内部标签。\n\n"
            "## 报告结构（固定 5 部分，必须按顺序）\n\n"
            "见 expected_output。\n\n"
            "⚠️ **输出约束（最高优先级）**：\n"
            "你的 Final Answer **必须只能**是 5 部分 Markdown 报告本身。"
            "**绝对禁止**在报告中回显任务描述中的任何指令性文字"
            "（如\"5 部分结构\"、\"URL 规则\"、\"修正清单\"、"
            "\"Markdown 格式\"、\"中文撰写\"、\"🚨 核心规则\" 等）。"
            "若你的输出包含上述指令词,该输出将被前端过滤器拒绝。\n\n"
            "报告末尾固定加上:\n"
            "`---`\n"
            "`> ⚠️ 本报告由 AI 自动生成，基于博查全网搜索的公开网页，仅供信息参考。`"
        ),
        expected_output=(
            "A Chinese markdown report with exactly these 5 sections in order:\n\n"
            "### 1. 执行摘要\n"
            "2-3 段，概述本期核心发现、最重要的趋势、最值得关注的风险。\n\n"
            "### 2. 核心主题\n"
            "每个主题写 8-12 句,叙事要具体、有细节。每个主题按以下结构展开:\n"
            "- 发生了什么（具体事件、参与方、时间、地点）\n"
            "- 背景是什么（为什么发生、行业/政策背景）\n"
            "- 意味着什么（影响分析、趋势信号、关联事件）\n"
            "- 有哪些值得关注的数据或细节\n"
            "提取 2-3 个最重要的主题,每个主题不少于 200 字。\n\n"
            "### 3. 风险与挑战\n"
            "针对核心主题，分析潜在风险: 市场接受度、技术成熟度、监管政策、竞争格局等。\n\n"
            "### 4. 关键结论\n"
            "3-5 条精炼结论，每条 1-2 句，可操作、有指向性。\n\n"
            "### 5. 参考来源\n"
            "编号列表，格式: 1. [标题](URL) — 公众号, YYYY-MM-DD\n"
            "末尾加上 --- 和 AI 免责声明。"
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
            "5. **报告质量** (权重 ×1.5): Is the fixed 5-section format followed? "
            "Are core themes detailed (≥200 words each, concrete narrative)? "
            "AI disclaimer present at bottom? "
            "Any example.com URLs or fabricated content?\n"
            "6. **数据流完整性** (权重 ×1.0): 逐篇追踪，制作文章对照表:\n"
            "   | Stage1编号 | 文章标题 | Stage2分析 | Stage3核查 | Stage4报告 | 备注 |\n"
            "   |-----------|---------|:---:|:---:|:---:|------|\n"
            "   ...\n"
            "   **注意: 允许同义改写和合理措辞调整，不要求逐字一致。**\n"
            "   仅关注实质信息是否丢失或严重扭曲（如数据错误、事件张冠李戴）。\n"
            "   URL 是否为真实链接（非 example.com）。\n\n"
            "## 评分注意事项\n"
            "- **日期缺失不扣分** — 搜索工具的限制，非 Agent 问题\n"
            "- **日期来自搜索结果就是正确的** — 搜索结果中返回的日期已经过工具"
            "按设定时间范围过滤，不要质疑日期的准确性\n"
            "- **搜索结果中的 URL 即真实链接** — 博查返回的是目标网站的直链，"
            "可以直接访问原文，不算占位符或无效链接\n"
            "- **不要无中生有** — 如果报告中确实引用了文章并提供了 URL，"
            "就不要声称\"报告未引用任何文章\"\n"
            "- **核心主题要具体** — 空洞概括 < 具体叙事\n"
            "- **报告格式固定 5 部分** — 多了少了都扣分\n\n"
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
