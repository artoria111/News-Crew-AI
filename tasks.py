from crewai import Task


def create_tasks(collector, analyzer, reporter):
    collect_task = Task(
        description=(
            "Search for the latest news articles about the topic: {topic}.\n"
            "Use RSSNewsSearchTool to find up to {max_articles} articles "
            "published within the last {days_back} days.\n"
            "For each article found:\n"
            "- Record the title, URL, source, and publication date\n"
            "- Use the ScrapeWebsiteTool to extract the full article content\n"
            "- Summarize the key points in 2-3 sentences in Chinese\n\n"
            "Compile everything into a structured collection with clear "
            "sections for each article. If no articles are found or a "
            "website cannot be scraped, note this clearly and continue "
            "with the available information."
        ),
        expected_output=(
            "A structured list of news articles. Each entry includes: "
            "title, source URL, publication date, source name, a 2-3 "
            "sentence Chinese summary, and 3-5 key bullet points "
            "extracted from the article content."
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
