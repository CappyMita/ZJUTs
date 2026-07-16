"""
RAG 系统核心引擎
===============
完整的 RAG 流程：
  加载文档 → 切分文本 → 向量化 → 存入向量库 → [用户提问] → 召回 → 重排序 → 生成回答

架构概览:
  DocumentProcessor  - 加载并切分文档
  EmbeddingEngine    - 将文本转为向量（本地 BGE 模型）
  VectorStoreManager - ChromaDB 向量库管理
  Reranker           - 对召回结果进行重排序（本地 Cross-Encoder）
  RAGEngine          - 编排整个流程

两阶段检索（召回 + 重排）：
  ┌─────────┐     ┌──────────────┐     ┌────────────┐
  │ 用户问题 │ ──► │ 召回 (Recall) │ ──► │ 重排 (Rerank)│ ──► 最相关片段
  └─────────┘     │ top_k=20     │     │ top_n=5    │
                  └──────────────┘     └────────────┘
                        ↑                      ↑
                  向量相似度搜索          Cross-Encoder 精排
"""

import os
import logging
from typing import List, Tuple, Optional

# !!! config 必须在 sentence_transformers 之前导入 !!!
# 因为 config 中设置了 HF_ENDPOINT 镜像环境变量，
# 需要在使用 HuggingFace 的任何库之前生效。
import config

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

import chromadb
from chromadb.config import Settings as ChromaSettings

from sentence_transformers import SentenceTransformer, CrossEncoder

from openai import OpenAI

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. 文档处理器：加载 + 切分
# ============================================================
class DocumentProcessor:
    """负责加载 Markdown 文档并切分为合适大小的 chunk。"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    def load_and_split(self, file_path: str) -> List[Document]:
        """
        加载 Markdown 文件，切分为 chunk 列表。

        参数:
            file_path: Markdown 文件路径

        返回:
            List[Document]: 每个 Document 包含 page_content 和 metadata
        """
        logger.info(f"📄 加载文档: {file_path}")

        # 使用 TextLoader 加载 Markdown（轻量，无需 unstructured 依赖）
        loader = TextLoader(file_path, encoding="utf-8")
        raw_docs = loader.load()
        logger.info(f"   原始文档片段数: {len(raw_docs)}")

        # 递归字符切分器 —— 按段落 → 换行 → 句号 → 字符的优先级切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )

        chunks = text_splitter.split_documents(raw_docs)
        logger.info(f"   切分后 chunk 数: {len(chunks)}")
        logger.info(f"   平均 chunk 长度: {sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} 字符")

        return chunks


# ============================================================
# 2. Embedding 引擎：文本 → 向量
# ============================================================
class EmbeddingEngine:
    """
    使用 BAAI/bge-small-zh-v1.5 模型将文本转为向量。
    该模型专为中文优化，体积小（~100MB），速度快。
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or config.EMBEDDING_MODEL_NAME
        logger.info(f"🔤 加载 Embedding 模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        # 兼容新旧版本 sentence-transformers
        try:
            self._dimension = self.model.get_sentence_embedding_dimension()
        except AttributeError:
            self._dimension = self.model.get_embedding_dimension()
        logger.info(f"   向量维度: {self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转为向量列表。

        参数:
            texts: 文本字符串列表

        返回:
            List[List[float]]: 对应的向量列表
        """
        # normalize_embeddings=True 使向量归一化，便于余弦相似度计算
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """将单个查询文本转为向量。"""
        return self.embed([query])[0]


# ============================================================
# 3. 向量库管理器：ChromaDB 存储与召回
# ============================================================
class VectorStoreManager:
    """
    基于 ChromaDB 的向量存储和相似度检索。

    ChromaDB 优势:
      - 轻量级，纯 Python 实现
      - 支持持久化存储
      - 内置多种相似度算法
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or config.VECTORDB_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = None

    def collection_exists(self, name: str = "arknights_guide") -> bool:
        """检查 collection 是否已存在。"""
        existing = self.client.list_collections()
        return any(c.name == name for c in existing)

    def create_collection(self, name: str = "arknights_guide", dimension: int = 512):
        """创建新的 collection（如果已存在则先删除）。"""
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )
        logger.info(f"📦 创建向量库 collection: {name} (维度={dimension})")

    def get_collection(self, name: str = "arknights_guide"):
        """获取已有的 collection。"""
        self.collection = self.client.get_collection(name)
        logger.info(f"📦 加载已有向量库: {name} (文档数={self.collection.count()})")

    def add_documents(self, chunks: List[Document], embeddings: List[List[float]]):
        """将文档块和对应向量存入 ChromaDB。"""
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"   已存入 {len(chunks)} 个文档块")

    def recall(self, query_embedding: List[float], top_k: int = None) -> Tuple[List[str], List[float]]:
        """
        【召回阶段】根据查询向量，从向量库中检索最相似的 top_k 个文档。

        使用余弦相似度进行语义级粗筛，速度快但精度有限。

        参数:
            query_embedding: 查询文本的向量
            top_k: 返回的候选数量

        返回:
            (documents, distances): 文档文本列表 和 对应距离列表
        """
        top_k = top_k or config.RECALL_TOP_K
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"],
        )
        docs = results["documents"][0]
        distances = results["distances"][0]
        logger.info(f"🔍 召回阶段: 获取 {len(docs)} 个候选片段")
        return docs, distances


# ============================================================
# 4. 重排序器：对召回结果精细排序
# ============================================================
class Reranker:
    """
    使用 BAAI/bge-reranker-v2-m3 Cross-Encoder 进行重排序。

    为什么需要重排序？
      - 向量召回（Embedding）做的是"粗筛"，速度快但精度不够
      - 重排序（Cross-Encoder）将 query 和 document 拼接后共同编码，
        能捕捉更精细的语义匹配关系，但速度较慢
      - 因此采用「先粗筛 20 条，再精排 5 条」的策略，兼顾速度与精度
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or config.RERANKER_MODEL_NAME
        logger.info(f"🎯 加载 Reranker 模型: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("   Reranker 模型加载完成")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None,
    ) -> Tuple[List[str], List[float]]:
        """
        【重排阶段】使用 Cross-Encoder 对候选文档逐一打分，按相关性降序排列。

        参数:
            query: 用户问题
            documents: 召回阶段返回的候选文档列表
            top_n: 最终保留的文档数量

        返回:
            (reranked_docs, scores): 重排后的文档列表 和 对应的相关性分数
        """
        top_n = top_n or config.RERANK_TOP_N

        if not documents:
            return [], []

        # 构造 (query, document) 对
        pairs = [[query, doc] for doc in documents]

        # Cross-Encoder 对每一对打分
        scores = self.model.predict(pairs, show_progress_bar=False)

        # 按分数降序排列
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked_docs = [doc for doc, _ in ranked[:top_n]]
        reranked_scores = [float(score) for _, score in ranked[:top_n]]

        logger.info(f"🎯 重排阶段: {len(documents)} 条候选 → 保留 {len(reranked_docs)} 条")
        if reranked_scores:
            logger.info(f"   最高分: {reranked_scores[0]:.4f}, 最低分: {reranked_scores[-1]:.4f}")

        return reranked_docs, reranked_scores


# ============================================================
# 5. RAG 引擎：编排完整流程
# ============================================================
class RAGEngine:
    """
    RAG 系统主引擎，编排所有组件。

    完整流程:
      1. 用户提问
      2. 问题向量化
      3. 召回（向量相似度搜索，粗筛 top_k=20）
      4. 重排（Cross-Encoder 精排，保留 top_n=5）
      5. 拼接上下文 + 问题，调用 DeepSeek 生成回答
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("🚀 初始化 RAG 系统")
        logger.info("=" * 60)

        # 初始化各组件
        self.embedder = EmbeddingEngine()
        self.reranker = Reranker()
        self.vector_store = VectorStoreManager()

        # DeepSeek 客户端（兼容 OpenAI 接口）
        self.llm_client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )

        # 尝试加载已有向量库
        self._init_vector_store()

        logger.info("=" * 60)
        logger.info("✅ RAG 系统初始化完成，可以开始提问！")
        logger.info("=" * 60)

    def _init_vector_store(self):
        """加载或创建向量库。"""
        if self.vector_store.collection_exists():
            self.vector_store.get_collection()
        else:
            logger.info("⚠️  向量库尚未构建，请先运行 setup.py 初始化数据")

    def is_ready(self) -> bool:
        """检查向量库是否已就绪。"""
        return self.vector_store.collection is not None

    def query(self, question: str, verbose: bool = True) -> dict:
        """
        执行一次完整的 RAG 查询。

        参数:
            question: 用户问题
            verbose: 是否打印详细日志

        返回:
            dict: {
                "question": 原始问题,
                "recalled_docs": 召回的文档列表,
                "recall_distances": 召回距离,
                "reranked_docs": 重排后的文档列表,
                "rerank_scores": 重排序分数,
                "answer": LLM 生成的回答,
            }
        """
        if not self.is_ready():
            return {"error": "向量库未就绪，请先运行 setup.py 构建向量库"}

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"❓ 用户问题: {question}")
            logger.info(f"{'='*60}")

        # ---- Step 1: 问题向量化 ----
        query_vec = self.embedder.embed_query(question)

        # ---- Step 2: 召回阶段 ----
        recalled_docs, distances = self.vector_store.recall(query_vec)

        # ---- Step 3: 重排序阶段 ----
        reranked_docs, rerank_scores = self.reranker.rerank(question, recalled_docs)

        # ---- Step 4: 生成回答 ----
        answer = self._generate(question, reranked_docs, verbose=verbose)

        return {
            "question": question,
            "recalled_docs": recalled_docs,
            "recall_distances": distances,
            "reranked_docs": reranked_docs,
            "rerank_scores": rerank_scores,
            "answer": answer,
        }

    def _generate(self, question: str, context_docs: List[str], verbose: bool = True) -> str:
        """
        调用 DeepSeek API 基于检索到的上下文生成回答。

        参数:
            question: 用户问题
            context_docs: 重排后的相关文档片段
            verbose: 是否打印详细日志

        返回:
            str: LLM 生成的回答
        """
        # 拼接上下文
        context = "\n\n---\n\n".join(
            [f"[参考片段 {i+1}]\n{doc}" for i, doc in enumerate(context_docs)]
        )

        # 构造 Prompt
        system_prompt = (
            "你是一个《明日方舟》游戏攻略助手。"
            "请严格根据下面提供的参考片段来回答问题。"
            "如果参考片段中找不到相关信息，请如实告知用户'当前攻略中未提及此内容'。"
            "回答要条理清晰，使用中文。"
        )

        user_prompt = f"""## 参考攻略内容

{context}

## 用户问题

{question}

请根据上面的攻略内容回答用户的问题："""

        if verbose:
            logger.info(f"🤖 调用 DeepSeek 生成回答...")
            logger.info(f"   System Prompt 长度: {len(system_prompt)} 字符")
            logger.info(f"   上下文总长度: {len(context)} 字符")

        try:
            response = self.llm_client.chat.completions.create(
                model=config.DEEPSEEK_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # 低温度保证回答稳定、忠于原文
                max_tokens=2048,
            )
            answer = response.choices[0].message.content

            if verbose:
                logger.info(f"   生成回答长度: {len(answer)} 字符")
                logger.info(f"   Token 用量: {response.usage}")

            return answer

        except Exception as e:
            logger.error(f"❌ DeepSeek API 调用失败: {e}")
            return f"抱歉，调用 AI 模型时出错: {str(e)}"


# ============================================================
# 构建向量库（首次运行的初始化脚本）
# ============================================================
def build_vector_store(source_file: str = None, data_file: str = None):
    """
    从零构建向量库:
      1. 自动搜索 Downloads 中的攻略文件并复制到 data/
      2. 加载并切分文档
      3. 向量化所有 chunk
      4. 存入 ChromaDB

    参数:
        source_file: 原始攻略文件路径（可选，不传则自动搜索 Downloads）
        data_file: 目标 data 目录中的文件路径
    """
    data_file = data_file or config.DATA_FILE

    logger.info("=" * 60)
    logger.info("🏗️  开始构建向量库")
    logger.info("=" * 60)

    # ---- Step 1: 自动搜索或使用指定的源文件 ----
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if source_file is None:
        # 自动在 Downloads 目录中搜索攻略文件
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        candidates = []
        for f in os.listdir(downloads):
            full = os.path.join(downloads, f)
            if os.path.isfile(full) and f.endswith(".md"):
                # 搜索包含"明日方舟"或"界园"关键词的 md 文件
                if "界园" in f or "明日方舟" in f or "攻略" in f:
                    candidates.append(full)
        if candidates:
            source_file = candidates[0]
            logger.info(f"📋 自动找到攻略文件: {os.path.basename(source_file)}")
        else:
            logger.error("❌ 未在 Downloads 中找到攻略 .md 文件")
            logger.error("   请手动指定 source_file 参数")
            return False

    if not os.path.exists(source_file):
        logger.error(f"❌ 源文件不存在: {source_file}")
        return False

    import shutil
    shutil.copy2(source_file, data_file)
    logger.info(f"📋 已复制攻略文件到: {data_file}")

    # ---- Step 2: 加载并切分文档 ----
    processor = DocumentProcessor()
    chunks = processor.load_and_split(data_file)

    # ---- Step 3: 向量化 ----
    embedder = EmbeddingEngine()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.embed(texts)

    # ---- Step 4: 存入 ChromaDB ----
    store = VectorStoreManager()
    store.create_collection(dimension=embedder.dimension)
    store.add_documents(chunks, embeddings)

    logger.info("=" * 60)
    logger.info(f"✅ 向量库构建完成！共 {len(chunks)} 个 chunk")
    logger.info("=" * 60)
    return True
