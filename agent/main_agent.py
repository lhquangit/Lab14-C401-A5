import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "this",
    "that",
    "là",
    "và",
    "của",
    "có",
    "cho",
    "với",
    "trong",
    "khi",
    "được",
    "về",
    "thì",
    "gì",
}


def _normalize_doc_id(stem: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return f"doc_{sanitized}"


def _tokenize(text: str) -> Set[str]:
    tokens = {t.lower() for t in _WORD_RE.findall(text)}
    return {t for t in tokens if len(t) > 1 and t not in _STOPWORDS}


def _is_metadata_line(line: str) -> bool:
    lowered = line.lower()
    return lowered.startswith(("source:", "department:", "effective date:", "access:"))


class MainAgent:
    """
    RAG Agent dùng tài liệu thật trong thư mục docs/.
    - Retrieval: lexical scoring có chuẩn hóa token.
    - Generation: gọi LLM nếu có OPENAI_API_KEY, fallback extractive nếu không có.
    - Output: answer, retrieved_ids, metadata.
    """

    def __init__(self, model: str = "gpt-4o-mini", docs_dir: str = "docs", top_k: int = 3):
        self.name = f"RAGAgent-{model}"
        self.model = model
        self.top_k = top_k
        self.docs = self._load_docs(Path(docs_dir))
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def _load_docs(docs_dir: Path) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for path in sorted(docs_dir.glob("*.txt")):
            raw_text = path.read_text(encoding="utf-8-sig")
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            if not lines:
                continue

            title = lines[0]
            body_lines = [
                line for line in lines[1:] if not line.startswith("===") and not _is_metadata_line(line)
            ]
            content = " ".join(body_lines)[:2500]
            blob = f"{title}\n{content}".lower()
            docs.append(
                {
                    "id": _normalize_doc_id(path.stem),
                    "file": path.name,
                    "title": title,
                    "content": content,
                    "blob": blob,
                    "token_set": _tokenize(blob),
                }
            )

        if not docs:
            raise FileNotFoundError(f"Không tìm thấy file .txt trong {docs_dir}")
        return docs

    @staticmethod
    def _score_doc(question: str, question_tokens: Set[str], doc: Dict[str, Any]) -> float:
        overlap = question_tokens.intersection(doc["token_set"])
        if not overlap:
            return 0.0
        min_overlap = 1 if len(question_tokens) <= 4 else 2
        if len(overlap) < min_overlap:
            return 0.0

        score = float(len(overlap))
        question_lower = question.lower().strip()
        if len(question_lower) > 12 and question_lower in doc["blob"]:
            score += 3.0
        return score

    async def retrieve(self, question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.top_k
        question_tokens = _tokenize(question)
        if not question_tokens:
            return []

        scored: List[Dict[str, Any]] = []
        for doc in self.docs:
            score = self._score_doc(question, question_tokens, doc)
            if score <= 0:
                continue
            scored.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "score": score,
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["id"]))
        return scored[:k]

    @staticmethod
    def _build_fallback_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return "Câu hỏi nằm ngoài phạm vi tài liệu nội bộ trong bộ dữ liệu này."

        question_tokens = _tokenize(question)
        best_line = ""
        best_score = 0
        for line in contexts[0]["content"].split("."):
            clean_line = line.strip()
            if not clean_line:
                continue
            score = len(question_tokens.intersection(_tokenize(clean_line)))
            if score > best_score:
                best_line = clean_line
                best_score = score

        if best_line:
            return best_line + "."
        return contexts[0]["content"][:240].strip() + "."

    async def generate(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not contexts:
            return {
                "answer": "Câu hỏi nằm ngoài phạm vi tài liệu nội bộ trong bộ dữ liệu này.",
                "metadata": {"model": self.model, "tokens_used": 0, "sources": [], "llm_mode": "no_context"},
            }

        if self.client is None:
            return {
                "answer": self._build_fallback_answer(question, contexts),
                "metadata": {
                    "model": self.model,
                    "tokens_used": 0,
                    "sources": [c["id"] for c in contexts],
                    "llm_mode": "fallback_no_api_key",
                },
            }

        context_text = "\n\n".join(
            f"[{c['id']}] {c['title']}\n{c['content'][:700]}" for c in contexts
        )
        prompt = (
            "You are a precise AI assistant. Use ONLY the provided context.\n"
            "If the answer is missing in the context, respond that it is out of scope.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\nAnswer:"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            message = response.choices[0].message.content or ""
            answer = message.strip()
            usage = response.usage.total_tokens if response.usage else 0
            return {
                "answer": answer,
                "metadata": {
                    "model": self.model,
                    "tokens_used": usage,
                    "sources": [c["id"] for c in contexts],
                    "llm_mode": "api",
                },
            }
        except Exception as exc:
            return {
                "answer": self._build_fallback_answer(question, contexts),
                "metadata": {
                    "model": self.model,
                    "tokens_used": 0,
                    "sources": [c["id"] for c in contexts],
                    "llm_mode": "fallback_on_error",
                    "error": str(exc),
                },
            }

    async def query(self, question: str) -> Dict[str, Any]:
        contexts = await self.retrieve(question)
        result = await self.generate(question, contexts)
        return {
            "answer": result["answer"],
            "retrieved_ids": [c["id"] for c in contexts],
            "metadata": result["metadata"],
        }


if __name__ == "__main__":
    agent = MainAgent()

    async def _test():
        questions = [
            "Tôi quên mật khẩu, phải làm gì?",
            "Ticket P1 có SLA xử lý bao lâu?",
            "Giá vàng hôm nay bao nhiêu?",
        ]
        for q in questions:
            resp = await agent.query(q)
            print("\nQ:", q)
            print("A:", resp["answer"][:160])
            print("Retrieved:", resp["retrieved_ids"])
            print("Metadata:", resp["metadata"])

    asyncio.run(_test())
