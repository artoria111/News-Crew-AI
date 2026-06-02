import os
from crewai import Agent, LLM


def create_llm():
    model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    return LLM(
        model=f"openai/{model_name}",
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
        temperature=0.3,
        max_tokens=8192,
    )


def create_agents(search_tool, scrape_tool, llm):
    collector = Agent(
        role="News Collector Specialist",
        goal=(
            "Search the web for the latest news articles about specified topics "
            "using DuckDuckGo and extract full article content for analysis."
        ),
        backstory=(
            "You are a seasoned digital news curator with a keen eye for "
            "relevant, timely information. You excel at finding diverse "
            "sources across the internet using the Baidu News search engine "
            "and extracting meaningful content from web pages. You prioritize "
            "credible sources and provide thorough coverage. You work in "
            "Chinese and English to ensure comprehensive results."
        ),
        tools=[search_tool, scrape_tool],
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
