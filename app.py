import streamlit as st
from main import run_crew, ConfigError

# ── page config ──────────────────────────────────────────
st.set_page_config(
    page_title="News Crew AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── style ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .main-subheader { color: #888; font-size: 0.9rem; margin-top: 0; }
    .report-container { border: 1px solid #e0e0e0; border-radius: 10px; padding: 2rem; background: #fafafa; }
</style>
""", unsafe_allow_html=True)

# ── session state ────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

# ── sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=64)
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
        help="从 RSS 源中筛选匹配的文章数量",
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

    # history
    if st.session_state.history:
        st.markdown("### 📚 历史记录")
        for i, item in enumerate(reversed(st.session_state.history)):
            label = f"{item['topic'][:20]} ({item['time']})"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.last_result = item["result"]
                st.session_state.last_topic = item["topic"]
                st.rerun()

# ── main area ────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">📰 News Crew AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subheader">智能新闻搜集 · 分析 · 报告</p>', unsafe_allow_html=True)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"数据源: 36氪 · 澎湃新闻")

st.markdown("---")

# load history result
if st.session_state.last_result and not start_btn:
    st.success(f"📋 上次报告: **{st.session_state.last_topic}**")
    st.markdown(st.session_state.last_result)

# ── run crew ─────────────────────────────────────────────
if start_btn:
    if not topic.strip():
        st.warning("请输入新闻主题")
    else:
        try:
            with st.status("🔄 正在执行新闻分析流程...", expanded=True) as status:
                st.write("🔍 **阶段 1/3** — 搜集新闻中...")
                st.write("📊 **阶段 2/3** — 分析整合中...")
                st.write("📝 **阶段 3/3** — 生成报告中...")

                result = run_crew(
                    topic=topic.strip(),
                    max_articles=max_articles,
                    days_back=days_back,
                )

                status.update(
                    label="✅ 分析完成！",
                    state="complete",
                    expanded=False,
                )

            report = result.raw

            st.success(f"### 📄 报告: {topic}")
            st.markdown(report)

            # save to history
            from datetime import datetime
            st.session_state.history.append({
                "topic": topic,
                "time": datetime.now().strftime("%H:%M"),
                "result": report,
            })
            st.session_state.last_result = report
            st.session_state.last_topic = topic

            # download
            st.download_button(
                label="📥 下载报告 (Markdown)",
                data=report,
                file_name=f"news_report_{topic[:10]}.md",
                mime="text/markdown",
            )

        except ConfigError as e:
            st.error(f"⚠️ 配置错误: {e}")
        except Exception as e:
            st.error(f"❌ 运行出错: {e}")

# ── footer ───────────────────────────────────────────────
st.markdown("---")
st.caption("Powered by CrewAI + DeepSeek + Streamlit")
