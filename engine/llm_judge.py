import asyncio
import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_RUBRIC_ACCURACY = (
    "Score 1-5 for factual correctness versus Ground Truth. "
    "5=identical facts, 4=mostly correct minor gap, 3=partially correct, "
    "2=mostly wrong, 1=completely incorrect."
)
_RUBRIC_FAITHFULNESS = (
    "Score 1-5 for how faithfully the answer uses context without hallucinating. "
    "5=zero hallucination, 1=fabricated facts."
)
_RUBRIC_RELEVANCY = (
    "Score 1-5 for how directly the answer addresses the question. "
    "5=fully on-point, 1=off-topic."
)

_SYSTEM_PROMPT = (
    "You are a strict and unbiased AI evaluator. "
    "Return only valid JSON, with no markdown."
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

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text) if len(t) > 1}


def _score_by_ratio(ratio: float) -> int:
    if ratio >= 0.85:
        return 5
    if ratio >= 0.65:
        return 4
    if ratio >= 0.45:
        return 3
    if ratio >= 0.2:
        return 2
    return 1


class MultiModelJudge:
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
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def _is_valid_score(value: Any) -> bool:
        return isinstance(value, int) and 1 <= value <= 5

    def _normalize_judge_payload(self, payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        normalized = {
            "accuracy_score": payload.get("accuracy_score"),
            "faithfulness_score": payload.get("faithfulness_score"),
            "relevancy_score": payload.get("relevancy_score"),
            "reasoning": payload.get("reasoning", ""),
            "model": model,
            "error": payload.get("error", False),
        }
        for key in ("accuracy_score", "faithfulness_score", "relevancy_score"):
            if not self._is_valid_score(normalized[key]):
                normalized[key] = None
                normalized["error"] = True
        return normalized

    @staticmethod
    def _heuristic_judge(question: str, answer: str, ground_truth: str, model: str) -> Dict[str, Any]:
        q_tokens = _tokenize(question)
        a_tokens = _tokenize(answer)
        gt_tokens = _tokenize(ground_truth)

        overlap_with_gt = len(a_tokens.intersection(gt_tokens))
        gt_ratio = overlap_with_gt / max(len(gt_tokens), 1)
        accuracy_score = _score_by_ratio(gt_ratio)

        question_ratio = len(a_tokens.intersection(q_tokens)) / max(len(q_tokens), 1)
        relevancy_score = _score_by_ratio(question_ratio)

        gt_is_oos = any(k in ground_truth.lower() for k in ("outside", "out of scope", "ngoài phạm vi"))
        ans_is_oos = any(k in answer.lower() for k in ("outside", "out of scope", "ngoài phạm vi"))
        if gt_is_oos:
            faithfulness_score = 5 if ans_is_oos else 2
        else:
            precision = overlap_with_gt / max(len(a_tokens), 1)
            faithfulness_score = _score_by_ratio(precision)

        return {
            "accuracy_score": accuracy_score,
            "faithfulness_score": faithfulness_score,
            "relevancy_score": relevancy_score,
            "reasoning": "Heuristic fallback judge used because API judge is unavailable.",
            "model": model,
            "error": False,
            "heuristic": True,
        }

    async def _call_single_judge(
        self, model: str, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        if self.client is None:
            return self._heuristic_judge(question, answer, ground_truth, model=model)

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
            raw = json.loads((response.choices[0].message.content or "").strip())
            return self._normalize_judge_payload(raw, model=model)
        except Exception as exc:
            return {
                "accuracy_score": None,
                "faithfulness_score": None,
                "relevancy_score": None,
                "reasoning": f"[ERROR] {exc}",
                "model": model,
                "error": True,
            }

    @staticmethod
    def _comparable_criteria(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        criteria = ["accuracy_score", "faithfulness_score", "relevancy_score"]
        return [
            c
            for c in criteria
            if isinstance(a.get(c), (int, float)) and isinstance(b.get(c), (int, float))
        ]

    def _has_conflict(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        comparable = self._comparable_criteria(a, b)
        if not comparable:
            return False
        return any(abs(a[c] - b[c]) > self.conflict_threshold for c in comparable)

    @staticmethod
    def _aggregate(judges: List[Dict[str, Any]]) -> Dict[str, Any]:
        criteria = ["accuracy_score", "faithfulness_score", "relevancy_score"]
        valid_judges = [
            j for j in judges if all(isinstance(j.get(c), (int, float)) for c in criteria)
        ]
        if not valid_judges:
            return {
                "avgs": {c: None for c in criteria},
                "final_score": 0.0,
                "used_judges": 0,
            }

        n = len(valid_judges)
        avgs = {c: round(sum(j[c] for j in valid_judges) / n, 2) for c in criteria}
        final_score = round(sum(avgs.values()) / len(avgs), 2)
        return {"avgs": avgs, "final_score": final_score, "used_judges": n}

    async def evaluate_multi_judge(
        self, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        judge_a, judge_b = await asyncio.gather(
            self._call_single_judge(self.primary_model, question, answer, ground_truth),
            self._call_single_judge(self.secondary_model, question, answer, ground_truth),
        )

        comparable = self._comparable_criteria(judge_a, judge_b)
        if comparable:
            agreements = [
                abs(judge_a[c] - judge_b[c]) <= self.conflict_threshold for c in comparable
            ]
            agreement_rate = round(sum(agreements) / len(agreements), 2)
        else:
            agreement_rate = 0.0

        all_judges = [judge_a, judge_b]
        tiebreaker = None
        if self._has_conflict(judge_a, judge_b):
            tiebreaker = await self._call_single_judge(
                self.tiebreaker_model, question, answer, ground_truth
            )
            all_judges.append(tiebreaker)

        aggregated = self._aggregate(all_judges)
        individual_scores = {
            self.primary_model: judge_a,
            self.secondary_model: judge_b,
        }
        if tiebreaker is not None:
            individual_scores["tiebreaker"] = tiebreaker

        judge_errors = sum(1 for j in all_judges if j.get("error"))
        reasoning = judge_a.get("reasoning") or judge_b.get("reasoning") or ""

        return {
            "individual_scores": individual_scores,
            "final_score": aggregated["final_score"],
            "agreement_rate": agreement_rate,
            "agreement_observed_criteria": len(comparable),
            "conflict_detected": tiebreaker is not None,
            "judge_errors": judge_errors,
            "reasoning": reasoning,
            "scoring_mode": "api" if self.client else "heuristic",
        }

    async def check_position_bias(
        self, question: str, response_a: str, response_b: str
    ) -> Dict[str, Any]:
        if self.client is None:
            return {
                "original_order_winner": None,
                "swapped_order_winner": None,
                "position_bias_detected": None,
                "reason": "OpenAI API key missing; cannot run model-based bias check.",
            }

        async def _pick_winner(resp_1: str, resp_2: str) -> str:
            prompt = (
                f"Question: {question}\n"
                f"Response A: {resp_1}\n"
                f"Response B: {resp_2}\n"
                "Return JSON only: {\"winner\": \"A\" or \"B\"}"
            )
            try:
                result = await self.client.chat.completions.create(
                    model=self.primary_model,
                    messages=[
                        {"role": "system", "content": "You are an impartial judge."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                payload = json.loads(result.choices[0].message.content or "{}")
                return payload.get("winner", "?")
            except Exception:
                return "?"

        winner_ab, winner_ba_raw = await asyncio.gather(
            _pick_winner(response_a, response_b),
            _pick_winner(response_b, response_a),
        )
        winner_ba = "B" if winner_ba_raw == "A" else ("A" if winner_ba_raw == "B" else "?")
        bias = None if "?" in (winner_ab, winner_ba) else winner_ab != winner_ba

        return {
            "original_order_winner": winner_ab,
            "swapped_order_winner": winner_ba,
            "position_bias_detected": bias,
        }


if __name__ == "__main__":
    judge = MultiModelJudge()

    async def _test():
        q = "Ticket P1 có SLA xử lý bao lâu?"
        a = "Ticket P1 có SLA xử lý và khắc phục trong 4 giờ."
        gt = "Ticket P1 có SLA xử lý và khắc phục trong 4 giờ."
        result = await judge.evaluate_multi_judge(q, a, gt)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(_test())
