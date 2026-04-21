import asyncio
import os
from typing import List, Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------
# Knowledge base khớp với synthetic_gen.py – dùng để Retrieval thật
# -----------------------------------------------------------------------
DOCS = [
    {
        "id": "doc_eval_01",
        "title": "Overview of AI Evaluation",
        "content": "AI evaluation measures quality with metrics such as correctness, latency, and cost.",
    },
    {
        "id": "doc_retrieval_02",
        "title": "Retrieval Metrics",
        "content": "Hit Rate@k checks whether any relevant document is in top-k. MRR rewards early correct ranking.",
    },
    {
        "id": "doc_runner_03",
        "title": "Async Runner",
        "content": "Batch runners should isolate per-case failures so one exception does not fail the full batch.",
    },
    {
        "id": "doc_judge_04",
        "title": "Multi Judge",
        "content": "Using two judges improves robustness by comparing scores and measuring agreement rate.",
    },
    {
        "id": "doc_gate_05",
        "title": "Release Gate",
        "content": "A release gate compares current metrics versus baseline and blocks release when thresholds are missed.",
    },
    {
        "id": "doc_cost_06",
        "title": "Cost and Tokens",
        "content": "Token usage should be tracked per request to optimize evaluation spend.",
    },
]


class MainAgent:
    """
    Agent thực hiện quy trình RAG (Retrieval-Augmented Generation) chuẩn.
    - Retrieval: keyword-based search trên DOCS corpus.
    - Generation: gọi OpenAI GPT để tổng hợp câu trả lời từ context.
    - Output: đúng schema runner.py cần (answer, retrieved_ids, metadata).
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.name = f"RAGAgent-{model}"
        self.model = model
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Stage 1: Retrieval – keyword scoring trên corpus nội bộ
    # ------------------------------------------------------------------
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tìm kiếm keyword-based: đếm số từ câu hỏi xuất hiện trong doc.
        Trả về top_k doc có điểm cao nhất.
        """
        question_tokens = set(question.lower().split())
        scored = []
        for doc in DOCS:
            combined = (doc["title"] + " " + doc["content"]).lower()
            score = sum(1 for token in question_tokens if token in combined)
            if score > 0:
                scored.append({"id": doc["id"], "content": doc["content"], "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Stage 2: Generation – gọi LLM với context đã tìm được
    # ------------------------------------------------------------------
    async def generate(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gọi OpenAI ChatCompletion. Nếu không có context, báo rõ out-of-scope.
        """
        if not contexts:
            return {
                "answer": "The answer is outside the provided evaluation documents.",
                "metadata": {
                    "model": self.model,
                    "tokens_used": 0,
                    "sources": [],
                },
            }

        context_text = "\n".join(
            f"[{c['id']}] {c['content']}" for c in contexts
        )
        prompt = (
            "You are a precise AI assistant. Answer the question using ONLY the information "
            "provided in the context below. If the answer is not in the context, say so clearly.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            return {
                "answer": answer,
                "metadata": {
                    "model": self.model,
                    "tokens_used": tokens,
                    "sources": [c["id"] for c in contexts],
                },
            }
        except Exception as exc:
            return {
                "answer": f"[ERROR] LLM call failed: {exc}",
                "metadata": {"model": self.model, "tokens_used": 0, "sources": [], "error": True},
            }

    # ------------------------------------------------------------------
    # EntryPoint: pipeline hoàn chỉnh – interface mà runner.py gọi
    # ------------------------------------------------------------------
    async def query(self, question: str) -> Dict[str, Any]:
        """
        Chạy full RAG pipeline và trả về schema chuẩn:
          - answer        : str
          - retrieved_ids : List[str]   ← dùng để tính Hit Rate / MRR
          - metadata      : Dict        ← tokens_used, model, sources
        """
        contexts = await self.retrieve(question)
        result = await self.generate(question, contexts)

        return {
            "answer": result["answer"],
            "retrieved_ids": [c["id"] for c in contexts],
            "metadata": result["metadata"],
        }


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = MainAgent()

    async def _test():
        questions = [
            "What does Hit Rate@k measure in retrieval evaluation?",
            "Why track token usage during evaluation?",
            "Who won the FIFA World Cup in 2010?",   # out-of-context
        ]
        for q in questions:
            print(f"\nQ: {q}")
            resp = await agent.query(q)
            print(f"  Answer       : {resp['answer'][:120]}")
            print(f"  Retrieved IDs: {resp['retrieved_ids']}")
            print(f"  Tokens used  : {resp['metadata']['tokens_used']}")

    asyncio.run(_test())
