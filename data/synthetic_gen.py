import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOCS_DIR = Path("docs")
OUTPUT_PATH = Path("data/golden_set.jsonl")
MIN_CASES = 50

def load_documents(docs_dir: Path) -> List[Dict]:
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append({
            "id": f"doc_{path.stem.lower()}",
            "file": path.name,
            "text": text
        })
    return docs

async def generate_cases_for_doc(doc: Dict, num_cases: int = 10) -> List[Dict]:
    """Sử dụng OpenAI để sinh câu hỏi từ tài liệu"""
    print(f"--- Generating {num_cases} cases for {doc['file']} ---")
    
    prompt = f"""
Bạn là một chuyên gia đánh giá RAG. Dựa trên tài liệu dưới đây, hãy tạo ra {num_cases} cặp (Câu hỏi, Câu trả lời).

Tài liệu [{doc['id']}]:
{doc['text']}

Yêu cầu các loại câu hỏi:
1. 'fact-check' (Easy): Hỏi trực tiếp thông tin có trong bài.
2. 'inference' (Medium): Cần suy luận nhẹ hoặc kết hợp 2 ý.
3. 'adversarial' (Hard): Cố tình lừa AI bỏ qua context hoặc tấn công prompt.
4. 'ambiguous' (Hard): Câu hỏi thiếu ngữ cảnh để xem AI có biết hỏi lại không.
5. 'out-of-context' (Hard): Hỏi về thứ không có trong bài (Answer phải là 'Tôi không tìm thấy thông tin này').

Định dạng JSON output:
{{
  "cases": [
    {{
      "question": "Câu hỏi...",
      "expected_answer": "Câu trả lời...",
      "expected_retrieval_ids": ["{doc['id']}"],
      "metadata": {{
        "type": "fact-check/inference/adversarial/ambiguous/out-of-context",
        "difficulty": "easy/medium/hard"
      }}
    }}
  ]
}}
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a data generator for RAG evaluation. Output valid JSON only."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("cases", [])
    except Exception as e:
        print(f"Error generating for {doc['file']}: {e}")
        return []

async def main():
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("No documents found!")
        return

    all_cases = []
    # Chia số lượng câu hỏi đều cho các tài liệu để đạt >= 50 cases
    cases_per_doc = (MIN_CASES // len(docs)) + 2 

    tasks = [generate_cases_for_doc(doc, cases_per_doc) for doc in docs]
    results = await asyncio.gather(*tasks)

    for cases in results:
        all_cases.extend(cases)

    # Đánh ID cho từng case
    for idx, case in enumerate(all_cases):
        case["id"] = f"case_{idx+1:03d}"

    # Lưu file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"✅ Đã tạo {len(all_cases)} cases trong {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
