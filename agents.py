import os
from crewai import Agent, LLM


def create_llm(provider: str = ""):
    provider = provider or os.getenv("LLM_PROVIDER", "siliconflow")

    providers = {
        "deepseek": {
            "model_name": os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1"),
        },
        "siliconflow": {
            "model_name": os.getenv("SILICONFLOW_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
            "api_key": os.getenv("SILICONFLOW_API_KEY"),
            "base_url": os.getenv("SILICONFLOW_URL", "https://api.siliconflow.cn/v1"),
        },
    }

    cfg = providers.get(provider, providers["siliconflow"])

    return LLM(
        model=f"openai/{cfg['model_name']}",
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.3,
        max_tokens=8192,
    )


def create_agents(search_tool, llm):
    collector = Agent(
        role="News Collector Specialist",
        goal=(
            "Search the web for the latest news articles about specified topics "
            "from RSS feeds and organize them into structured collections."
        ),
        backstory=(
            "You are a seasoned digital news curator with a keen eye for "
            "relevant, timely information. You search across 8 RSS feeds "
            "covering Chinese and international sources, filter by topic "
            "relevance, and organize concise article summaries. You work "
            "in Chinese and English to ensure comprehensive results."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    analyzer = Agent(
        role="News Analysis Expert",
        goal=(
            "Analyze collected news articles to identify cross-cutting themes, "
            "patterns, biases, and key insights. Provide balanced, structured analysis."
        ),
        backstory=(
            "You are a senior data analyst specializing in media content "
            "analysis. You can identify trends, conflicting viewpoints, and "
            "key takeaways across multiple news sources. You excel at "
            "connecting dots across disparate articles and providing "
            "nuanced analysis that accounts for source diversity. "
            "You present your analysis in Chinese."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Senior News Reporter",
        goal=(
            "Create comprehensive, well-structured markdown reports that present "
            "analyzed news findings in a clear, engaging, and professional format."
        ),
        backstory=(
            "You are an award-winning journalist with decades of experience "
            "in technology and business reporting. You specialize in "
            "transforming complex information into clear, compelling "
            "narratives. Your reports are known for their clarity, accuracy, "
            "and readability. You always include proper source attribution "
            "and an executive summary. You write reports in Chinese."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return collector, analyzer, reporter


def create_auditor(llm):
    auditor = Agent(
        role="Quality Assurance Auditor",
        goal=(
            "Rigorously evaluate the quality, accuracy, and efficiency of "
            "the news collection, analysis, and reporting process across "
            "multiple dimensions. Produce a structured scorecard with "
            "actionable feedback."
        ),
        backstory=(
            "You are a veteran editorial quality auditor with 20 years of "
            "experience in newsroom quality control. You have a sharp eye "
            "for detecting inaccuracies, biases, logical gaps, and "
            "information loss in multi-stage editorial workflows. You are "
            "known for fair but rigorous evaluations that help teams "
            "continuously improve. You evaluate work systematically across "
            "multiple dimensions and produce structured scorecards with "
            "specific, actionable feedback in Chinese."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    return auditor


def create_fact_checker(llm):
    return Agent(
        role="Fact Checker & Verification Specialist",
        goal=(
            "Cross-verify every key claim, fact, and data point in the analysis "
            "against the original collected articles. Identify and flag any "
            "distortions, misinterpretations, or fabricated claims."
        ),
        backstory=(
            "You are a meticulous fact-checker with 15 years of experience in "
            "investigative journalism. You never trust a secondary source — you "
            "always go back to the original material. You are known for catching "
            "subtle misinterpretations that others miss. A claim like 'X company "
            "launched Y division' must be verified against what the original "
            "article actually said. You flag every discrepancy, no matter how "
            "small, and provide the correct interpretation with direct quotes "
            "from the source. You work in Chinese."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
