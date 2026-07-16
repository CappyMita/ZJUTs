"""
数据初始化脚本
=============
首次运行 RAG 系统前需要执行此脚本，完成：
  1. 复制攻略 Markdown 文件到 data/ 目录
  2. 切分文档为 chunk
  3. 向量化并存入 ChromaDB

运行方式:
  python setup.py
"""

import sys
import os

# 确保能找到 config 和 rag_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import build_vector_store

if __name__ == "__main__":
    success = build_vector_store()
    if success:
        print("\n[OK] 初始化成功！现在可以运行 main.py 开始提问了。")
    else:
        print("\n[FAIL] 初始化失败，请检查错误信息。")
