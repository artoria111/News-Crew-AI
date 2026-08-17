import re

import streamlit as st

from main import ConfigError, LLM_PROVIDER, SEARCH_PROVIDER, run_crew


def clean_md(text: str) -> str:
    """strip raw html tags that leak into markdown rendering"""
    return re.sub(r"<br\s*/?>", "", text, flags=re.IGNORECASE)


# ── page config ──────────────────────────────────────────
st.set_page_config(
    page_title="News Crew AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── style ────────────────────────────────────────────────
st.markdown(
    """
<style>
    html { font-size: 18px; }
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .main-subheader { color: #888; font-size: 1rem; margin-top: 0; }
    .stMarkdown, .stText, p, li { font-size: 1.05rem; }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.6rem !important; }
    h3 { font-size: 1.3rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── session state ────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "last_audit" not in st.session_state:
    st.session_state.last_audit = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

# ── sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 配置参数")

    topic = st.text_input(
        "📝 新闻主题",
        value="AI最新进展",
        placeholder="输入关键词，如：人工智能、新能源、芯片...",
    )

    max_articles = st.slider(
        "📊 文章数量",
        min_value=1,
        max_value=20,
        value=5,
        help=f"从 {SEARCH_PROVIDER} 结果中筛选的文章数量",
    )

    days_back = st.selectbox(
        "📅 时间范围",
        options=[1, 3, 7, 14, 30],
        index=2,
        format_func=lambda d: f"最近 {d} 天" if d > 1 else "今天",
    )

    st.markdown("---")

    start_btn = st.button(
        "🚀 开始分析",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")

    if st.session_state.history:
        st.markdown("### 📚 历史记录")
        for i, item in enumerate(reversed(st.session_state.history)):
            label = f"{item['topic'][:20]} ({item['time']})"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.last_report = item["report"]
                st.session_state.last_audit = item["audit"]
                st.session_state.last_topic = item["topic"]
                st.rerun()

# ── main area header ─────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">📰 News Crew AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="main-subheader">智能新闻搜集 · 分析 · 事实核查 · 报告 · 审核</p>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"数据源: {SEARCH_PROVIDER}")

st.markdown("---")

# ── show history result ──────────────────────────────────
if st.session_state.last_report and not start_btn:
    st.success(f"📋 上次报告: **{st.session_state.last_topic}**")
    st.markdown(clean_md(st.session_state.last_report))
    if st.session_state.last_audit:
        with st.expander("📊 查看质量审核评分卡", expanded=False):
            st.markdown(clean_md(st.session_state.last_audit))

# ── run crew ─────────────────────────────────────────────
if start_btn:
    if not topic.strip():
        st.warning("请输入新闻主题")
    else:
        try:
            with st.status("🔄 正在执行新闻分析流程...", expanded=True) as status:
                st.write("🔍 **阶段 1/5** — 搜集新闻中...")
                st.write("📊 **阶段 2/5** — 分析整合中...")
                st.write("🔬 **阶段 3/5** — 事实核查中...")
                st.write("📝 **阶段 4/5** — 生成报告中...")
                st.write("🛡️ **阶段 5/5** — 质量审核中...")

                report, audit = run_crew(
                    topic=topic.strip(),
                    max_articles=max_articles,
                    days_back=days_back,
                )

                status.update(
                    label="✅ 全部分析完成！",
                    state="complete",
                    expanded=False,
                )

            # report
            report = clean_md(report)
            st.success(f"### 📄 报告: {topic}")
            st.markdown(report)

            st.download_button(
                label="📥 下载报告 (Markdown)",
                data=report,
                file_name=f"news_report_{topic[:10]}.md",
                mime="text/markdown",
            )

            # audit scorecard
            if audit:
                audit = clean_md(audit)
                st.markdown("---")
                with st.expander("📊 质量审核评分卡", expanded=True):
                    st.markdown(audit)
                    st.download_button(
                        label="📥 下载审核评分卡",
                        data=audit,
                        file_name=f"audit_{topic[:10]}.md",
                        mime="text/markdown",
                    )

                # save to history
                from datetime import datetime

                st.session_state.history.append(
                    {
                        "topic": topic,
                        "time": datetime.now().strftime("%H:%M"),
                        "report": report,
                        "audit": audit,
                    }
                )
                st.session_state.last_report = report
                st.session_state.last_audit = audit
                st.session_state.last_topic = topic

        except ConfigError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"❌ 运行出错: {e}")

# ── footer ───────────────────────────────────────────────
st.markdown("---")
st.caption(f"Powered by CrewAI + Streamlit | 搜索: {SEARCH_PROVIDER.split(' ')[0]} | LLM: {LLM_PROVIDER}")
