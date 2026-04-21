import asyncio
import json
import os
from typing import Any, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------
# Rubric definitions – embedded into every judge prompt
# -----------------------------------------------------------------------
_RUBRIC_ACCURACY = (
    "Score 1-5 for factual correctness versus Ground Truth. "
    "5=identical facts, 4=mostly correct minor gap, "
    "3=partially correct, 2=mostly wrong, 1=completely incorrect."
)
_RUBRIC_FAITHFULNESS = (
    "Score 1-5 for how faithfully the answer uses the context without hallucinating. "
    "5=zero hallucination, 1=answer fabricates facts not in context."
)
_RUBRIC_RELEVANCY = (
    "Score 1-5 for how directly the answer addresses the question. "
    "5=fully on-point, 1=completely off-topic."
)

_SYSTEM_PROMPT = (
    "You are a strict, unbiased AI evaluation judge. "
    "Respond ONLY with a valid JSON object — no markdown, no commentary."
)

_JUDGE_PROMPT = """\
Evaluate the AI answer using the three rubrics below.

QUESTION:
{question}

AI ANSWER:
{answer}

GROUND TRUTH:
{ground_truth}

RUBRICS:
1. accuracy     : {rubric_accuracy}
2. faithfulness : {rubric_faithfulness}
3. relevancy    : {rubric_relevancy}

Return EXACTLY this JSON:
{{
  "accuracy_score": <int 1-5>,
  "faithfulness_score": <int 1-5>,
  "relevancy_score": <int 1-5>,
  "reasoning": "<one concise sentence>"
}}"""


class MultiModelJudge:
    """
    Multi-model consensus judge cho Lab 14 AI Evaluation Factory.

    Quy trình:
    1. Gọi song song primary_model (gpt-4o) và secondary_model (gpt-4o-mini).
    2. Tính Agreement Rate: tỉ lệ tiêu chí 2 judge chênh lệch ≤ conflict_threshold.
    3. Nếu có conflict (bất kỳ tiêu chí nào lệch > threshold) → gọi tie-breaker,
       trung bình 3 kết quả.
    4. Output chuẩn để runner.py đọc:
       individual_scores, final_score, agreement_rate, reasoning.
    """

    def __init__(
        self,
        primary_model: str = "gpt-4o",
        secondary_model: str = "gpt-4o-mini",
        tiebreaker_model: str = "gpt-4o",
        conflict_threshold: int = 1,
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.tiebreaker_model = tiebreaker_model
        self.conflict_threshold = conflict_threshold
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rubrics = {
            "accuracy": _RUBRIC_ACCURACY,
            "faithfulness": _RUBRIC_FAITHFULNESS,
            "relevancy": _RUBRIC_RELEVANCY,
        }

    # ------------------------------------------------------------------
    # Internal: call a single judge model
    # ------------------------------------------------------------------
    async def _call_single_judge(
        self, model: str, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            rubric_accuracy=_RUBRIC_ACCURACY,
            rubric_faithfulness=_RUBRIC_FAITHFULNESS,
            rubric_relevancy=_RUBRIC_RELEVANCY,
        )
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as exc:
            # Safe fallback: không crash cả batch khi 1 judge lỗi
            return {
                "accuracy_score": 0,
                "faithfulness_score": 0,
                "relevancy_score": 0,
                "reasoning": f"[ERROR] {exc}",
                "error": True,
            }

    # ------------------------------------------------------------------
    # Internal: check if two judge results have any conflicting criterion
    # ------------------------------------------------------------------
    def _has_conflict(self, a: Dict, b: Dict) -> bool:
        criteria = ["accuracy_score", "faithfulness_score", "relevancy_score"]
        return any(abs(a.get(c, 0) - b.get(c, 0)) > self.conflict_threshold for c in criteria)

    # ------------------------------------------------------------------
    # Internal: compute average scores and agreement rate from 2+ judges
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate(judges: list) -> Dict[str, Any]:
        criteria = ["accuracy_score", "faithfulness_score", "relevancy_score"]
        n = len(judges)
        avgs = {c: sum(j.get(c, 0) for j in judges) / n for c in criteria}
        final_score = round(sum(avgs.values()) / len(avgs), 2)
        return {"avgs": avgs, "final_score": final_score}

    # ------------------------------------------------------------------
    # Public: Multi-Judge consensus entry point (called by runner.py)
    # ------------------------------------------------------------------
    async def evaluate_multi_judge(
        self, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        """
        Gọi ít nhất 2 judge song song. Có logic xử lý conflict tự động.

        Returns:
            individual_scores : dict – điểm từng model
            final_score       : float – điểm tổng hợp cuối (1–5)
            agreement_rate    : float – tỉ lệ tiêu chí đồng thuận (0–1)
            reasoning         : str   – giải thích từ primary judge
        """
        # --- Bước 1: Gọi đồng thời 2 model ---
        judge_a, judge_b = await asyncio.gather(
            self._call_single_judge(self.primary_model, question, answer, ground_truth),
            self._call_single_judge(self.secondary_model, question, answer, ground_truth),
        )

        # --- Bước 2: Tính Agreement Rate (từ 2 judge gốc) ---
        criteria = ["accuracy_score", "faithfulness_score", "relevancy_score"]
        agreements = [
            abs(judge_a.get(c, 0) - judge_b.get(c, 0)) <= self.conflict_threshold
            for c in criteria
        ]
        agreement_rate = round(sum(agreements) / len(agreements), 2)

        # --- Bước 3: Xử lý conflict – gọi tie-breaker nếu cần ---
        all_judges = [judge_a, judge_b]
        tiebreaker = None
        if self._has_conflict(judge_a, judge_b):
            tiebreaker = await self._call_single_judge(
                self.tiebreaker_model, question, answer, ground_truth
            )
            all_judges.append(tiebreaker)

        # --- Bước 4: Tổng hợp final score từ tất cả judge ---
        aggregated = self._aggregate(all_judges)

        individual_scores = {
            self.primary_model: judge_a,
            self.secondary_model: judge_b,
        }
        if tiebreaker:
            individual_scores["tiebreaker"] = tiebreaker

        return {
            "individual_scores": individual_scores,
            "final_score": aggregated["final_score"],
            "agreement_rate": agreement_rate,
            "conflict_detected": tiebreaker is not None,
            "reasoning": judge_a.get("reasoning", ""),
        }

    # ------------------------------------------------------------------
    # Advanced: Position Bias detection (swap order, compare winner)
    # ------------------------------------------------------------------
    async def check_position_bias(
        self, question: str, response_a: str, response_b: str
    ) -> Dict[str, Any]:
        """
        Swap vị trí A/B để kiểm tra judge có bị thiên vị không.
        """
        async def _pick_winner(resp_1: str, resp_2: str) -> str:
            prompt = (
                f"Question: {question}\n"
                f"Response A: {resp_1}\nResponse B: {resp_2}\n"
                "Which response is better? Reply JSON: {\"winner\": \"A\" or \"B\"}"
            )
            try:
                r = await self.client.chat.completions.create(
                    model=self.primary_model,
                    messages=[
                        {"role": "system", "content": "You are an impartial judge."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                return json.loads(r.choices[0].message.content).get("winner", "?")
            except Exception:
                return "?"

        winner_ab, winner_ba_raw = await asyncio.gather(
            _pick_winner(response_a, response_b),
            _pick_winner(response_b, response_a),   # positions swapped
        )
        # Normalize: if judge picked "A" in swapped order, real winner is "B"
        winner_ba = "B" if winner_ba_raw == "A" else ("A" if winner_ba_raw == "B" else "?")

        return {
            "original_order_winner": winner_ab,
            "swapped_order_winner": winner_ba,
            "position_bias_detected": winner_ab != winner_ba,
        }


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    judge = MultiModelJudge()

    async def _test():
        print("=== Testing MultiModelJudge ===")
        q = "What does Hit Rate@k measure in retrieval evaluation?"
        a = "Hit Rate@k checks if any relevant document appears within the top-k results."
        gt = "It measures whether at least one relevant document appears in the top-k retrieved results."

        result = await judge.evaluate_multi_judge(q, a, gt)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(_test())
