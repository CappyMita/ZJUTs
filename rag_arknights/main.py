"""
RAG 系统交互入口
===============
启动命令行交互式问答，支持以下命令：
  - 直接输入问题：执行 RAG 查询
  - !detail：显示上一轮检索详情（召回了哪些片段、重排分数等）
  - !quit 或 !exit：退出程序

运行方式:
  python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import RAGEngine


def print_separator(char="─", length=60):
    print(char * length)


def print_retrieval_detail(result: dict):
    """打印检索详情：展示召回和重排的全过程。"""
    if "error" in result:
        print(f"❌ {result['error']}")
        return

    print_separator("─")
    print("📊 检索详情")
    print_separator("─")

    # 召回阶段
    print(f"\n🔍 【阶段一：召回】(向量相似度搜索，top_k=20)")
    print(f"   共召回 {len(result['recalled_docs'])} 条候选片段")
    for i, doc in enumerate(result["recalled_docs"][:5]):
        print(f"   [{i+1}] 距离={result['recall_distances'][i]:.4f} | {doc[:80]}...")
    if len(result["recalled_docs"]) > 5:
        print(f"   ... 还有 {len(result['recalled_docs']) - 5} 条")

    # 重排阶段
    print(f"\n🎯 【阶段二：重排序】(Cross-Encoder 精排，保留 top_n=5)")
    print(f"   最终保留 {len(result['reranked_docs'])} 条最相关片段")
    for i, (doc, score) in enumerate(zip(result["reranked_docs"], result["rerank_scores"])):
        print(f"   [{i+1}] 得分={score:.4f}")
        print(f"        {doc[:150]}...")
        print()

    print_separator("─")


def main():
    print("=" * 60)
    print("   《明日方舟》集成战略「岁的界园志异」攻略 RAG 系统")
    print("=" * 60)
    print()
    print("   📖 基于 DeepSeek + ChromaDB + BGE 模型")
    print("   🔍 两阶段检索：召回(粗筛) → 重排(精排)")
    print()
    print("   命令:")
    print("     直接输入问题 -- 查询攻略")
    print("     !detail      -- 显示上一轮检索详情")
    print("     !quit        -- 退出")
    print()

    # 初始化 RAG 引擎
    try:
        rag = RAGEngine()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("   请先运行: python setup.py")
        return

    if not rag.is_ready():
        print("⚠️  向量库尚未构建，请先运行: python setup.py")
        return

    # 保存上一次查询结果，用于 !detail 命令
    last_result = None

    # 交互循环
    while True:
        try:
            user_input = input("\n💬 请输入问题 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input in ("!quit", "!exit", "!q"):
            print("👋 再见！")
            break

        if user_input == "!detail":
            if last_result:
                print_retrieval_detail(last_result)
            else:
                print("⚠️  还没有查询记录，请先提问。")
            continue

        # 执行 RAG 查询
        print_separator("─")
        result = rag.query(user_input, verbose=True)
        last_result = result

        print_separator("─")
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"\n📝 【回答】\n")
            print(result["answer"])
            print()

        print_separator("─")
        print("💡 输入 !detail 查看检索详情 | !quit 退出")


if __name__ == "__main__":
    main()
