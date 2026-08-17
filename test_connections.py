"""独立连接测试脚本 — 不依赖 Streamlit,直接验证 LLM 与搜索 API。

用法: .venv/Scripts/python.exe test_connections.py [llm|bocha|all]
"""
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"


def test_llm() -> None:
    key = os.getenv("LLM_API_KEY")
    base = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
    print(f"[LLM] base={base} model={model} key={'***' if key else 'MISSING'}")
    if not key:
        print("[LLM] FAIL — .env 缺少 LLM_API_KEY")
        return

    from agents import create_llm

    llm = create_llm()
    t0 = time.time()
    try:
        resp = llm.call(
            messages=[{"role": "user", "content": "回复: 连接测试成功。请用一句话介绍你自己。"}]
        )
        print(f"[LLM] OK ({time.time() - t0:.1f}s)")
        print(f"[LLM] REPLY: {resp[:200]}")
    except Exception as e:
        print(f"[LLM] FAIL ({time.time() - t0:.1f}s)")
        print(f"[LLM] {type(e).__name__}: {str(e)[:400]}")


def test_bocha() -> None:
    key = os.getenv("BOCHA_API_KEY")
    base = os.getenv("BOCHA_BASE_URL", "https://api.bochaai.com/v1")
    print(f"[BOCHA] base={base} key={'***' if key else 'MISSING'}")
    if not key:
        print(
            "[BOCHA] SKIP — 请在 https://open.bochaai.com 免费注册后,"
            " 将 BOCHA_API_KEY 写入 .env, 然后重新运行本脚本"
        )
        return

    from tools import BochaSearchTool

    tool = BochaSearchTool(days_back=7, max_results=3)
    t0 = time.time()
    raw = tool._run("AI最新进展")
    print(f"[BOCHA] OK ({time.time() - t0:.1f}s), 返回 {len(raw)} 字符")
    print(raw[:800])


if __name__ == "__main__":
    if WHICH in ("all", "llm"):
        test_llm()
    if WHICH in ("all", "bocha"):
        print()
        test_bocha()
