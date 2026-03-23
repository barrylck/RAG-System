from dotenv import load_dotenv
load_dotenv()

import json
import anthropic
import sys
import os

# add parent directory to path so we can import query.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from query import query_rag

claude_client = anthropic.Anthropic()


def grade_answer(question, expected_answer, actual_answer):
    """Use Claude to grade the RAG system's answer on a 1-5 scale."""

    is_no_answer = expected_answer.strip().lower() == "not in documents"

    if is_no_answer:
        grading_prompt = f"""You are grading a RAG system's answer to a question that has NO answer in the documents.

Question: {question}
Actual Answer: {actual_answer}

The correct behaviour is for the system to say it could not find the information.
Score on a 1-5 scale:
5 - Correctly says the information is not in the documents
3 - Partially hedges but still attempts an answer
1 - Confidently gives an answer (hallucination)

Reply with ONLY a JSON object, nothing else. Example: {{"score": 4, "reason": "brief explanation"}}"""

    else:
        grading_prompt = f"""You are grading a RAG system's answer. Score it from 1 to 5.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Scoring criteria:
5 - Correct and complete
4 - Mostly correct, minor details missing
3 - Partially correct
2 - Mostly incorrect but contains some relevant info
1 - Completely wrong or hallucinated

Reply with ONLY a JSON object, nothing else. Example: {{"score": 4, "reason": "brief explanation"}}"""

    # retry up to 3 times in case Claude returns empty or malformed response
    for attempt in range(3):
        response = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": grading_prompt}]
        )

        raw = response.content[0].text.strip()

        if not raw:
            print(f"     [Warning] Empty response from grader, retrying ({attempt+1}/3)...")
            continue

        try:
            # strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            return result["score"], result["reason"]
        except json.JSONDecodeError:
            print(f"     [Warning] Could not parse grader response, retrying ({attempt+1}/3)...")
            print(f"     Raw response: {raw}")
            continue

    # fallback if all retries fail
    return 0, "grader failed to return valid JSON after 3 attempts"


def run_evaluation():
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    test_set_path = os.path.join(eval_dir, "test_set.json")

    with open(test_set_path, "r") as f:
        test_cases = json.load(f)

    results = []
    print(f"Running evaluation on {len(test_cases)} questions...\n")

    for test in test_cases:
        print(f"[{test['id']:02d}] [{test['category'].upper()}] {test['question']}")

        actual_answer = query_rag(test["question"], k=5)

        score, reason = grade_answer(
            test["question"],
            test["expected_answer"],
            actual_answer
        )

        result = {
            "id": test["id"],
            "category": test["category"],
            "question": test["question"],
            "expected": test["expected_answer"],
            "actual": actual_answer,
            "score": score,
            "reason": reason
        }
        results.append(result)
        print(f"     Score: {score}/5 — {reason}\n")

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"Overall average:  {avg_score:.2f} / 5.00")
    print(f"Total questions:  {len(results)}\n")

    print("By category:")
    categories = sorted(set(r["category"] for r in results))
    for category in categories:
        cat_results = [r for r in results if r["category"] == category]
        cat_avg = sum(r["score"] for r in cat_results) / len(cat_results)
        bar = "█" * int(cat_avg) + "░" * (5 - int(cat_avg))
        print(f"  {category:<16} {bar}  {cat_avg:.2f}/5  ({len(cat_results)} questions)")

    print("\nWORST FAILURES (bottom 3):")
    print("-" * 60)
    failures = sorted(results, key=lambda x: x["score"])[:3]
    for f in failures:
        print(f"Q:        {f['question']}")
        print(f"Expected: {f['expected']}")
        print(f"Actual:   {f['actual']}")
        print(f"Score:    {f['score']}/5 — {f['reason']}\n")

    output_path = os.path.join(eval_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to eval/eval_results.json")


if __name__ == "__main__":
    run_evaluation()