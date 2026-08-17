# 📰 News Crew AI

基于 [CrewAI](https://crewai.com) 的智能新闻搜集、分析、核查与报告系统。通过博查全网搜索获取多源新闻，五个 AI Agent 流水线协作，自动生成结构化的深度分析报告。

## ✨ 功能特性

- **全网新闻搜集** — 博查搜索覆盖中文全网网页，支持时效过滤 (天/周/月)，单次搜索命中多个不同网站信源
- **五 Agent 流水线** — 搜集 → 分析 → 事实核查 → 报告 → 审计，全自动协作
- **条件修正循环** — 事实核查发现错误时自动触发修正轮，确保报告准确性
- **质量过滤** — 自动筛除讲座预告、广告、招聘等低质文章
- **Web 前端** — Streamlit 构建，侧栏配置参数，主区实时展示报告与审计评分

## 🏗️ 架构

```
用户输入 (主题)
       │
       ▼
┌─────────────────────────────────────────────┐
│  Stage 1: News Collector (搜集员)            │
│  工具: 博查全网搜索 (Bocha Web Search)        │
│  输出: 结构化文章列表 (标题/URL/来源/日期/摘要) │
├─────────────────────────────────────────────┤
│  Stage 2: News Analyzer (分析员)             │
│  输出: 主题提炼 / 趋势识别 / 风险信号         │
├─────────────────────────────────────────────┤
│  Stage 3: Fact Checker (事实核查员)           │
│  对照原文逐条核实，标记错误和编造内容          │
├─────────────────────────────────────────────┤
│  Stage 4: Reporter (报告员)                   │
│  固定 5 部分: 摘要 / 核心主题 / 风险 / 结论 / 参考来源 │
│  ↓ (错误 ≥ 1 → 自动修正轮 → 再核查)           │
├─────────────────────────────────────────────┤
│  Stage 5: Auditor (审核员)                    │
│  6 维度加权评分 + 亮点/改进建议                │
└─────────────────────────────────────────────┘
       │
       ▼
  Streamlit 前端展示
  (报告 + 审计评分卡 + 下载)
```

## 🚀 快速开始

### 环境要求

- Python ≥ 3.12
- 任意兼容 OpenAI 格式的 API (DeepSeek / OpenAI / SiliconFlow / 智谱 / 通义千问 等)

### 安装

```bash
git clone https://github.com/AnonLee/NewsCrewAI.git
cd NewsCrewAI
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env` 并填入任意 OpenAI 兼容 API：

```env
LLM_API_KEY=sk-your-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL_NAME=deepseek-chat
BOCHA_API_KEY=sk-your-bocha-key
BOCHA_BASE_URL=https://api.bochaai.com/v1
```

只需改三个变量即可适配任意 LLM 厂商；`BOCHA_API_KEY` 在 [博查开放平台](https://open.bochaai.com) 免费注册获取。常用示例：

| 厂商 | LLM_BASE_URL | LLM_MODEL_NAME |
|------|-------------|----------------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| SiliconFlow | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` |

### 启动

```bash
streamlit run app.py
```

浏览器自动打开，在左侧栏输入主题、选择文章数量、点击"开始分析"即可。

## 📁 项目结构

```
├── app.py              # Streamlit Web 前端
├── main.py             # 流水线编排 (Phase 1/2/3)
├── agents.py           # 5 个 Agent 定义
├── tasks.py            # 所有 Task 定义 + 约束规则
├── tools.py            # 搜索工具 (博查全网搜索 + 搜狗微信备用)
├── pyproject.toml      # 依赖配置
├── requirements.txt    # 依赖配置
├── .env.example        # 环境变量模板
├── .gitignore
└── README.md
```

## 🔧 依赖

| 包 | 用途 |
|---|---|
| `crewai` | Agent / Task / Crew 核心框架 |
| `streamlit` | Web 前端 |
| `feedparser` | RSS 解析 |
| `beautifulsoup4` | HTML 解析 |
| `requests` | HTTP 请求 |
| `python-dotenv` | 环境变量管理 |

## ⚠️ 免责声明

本系统由 AI 自动生成报告，基于博查全网搜索的公开网页，仅供信息参考，不构成任何建议。报告中可能存在遗漏或偏差，请读者自行核实关键信息。

## 📄 License

MIT License.
