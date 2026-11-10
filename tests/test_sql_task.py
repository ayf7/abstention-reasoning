"""Unit tests for SqlTask."""

import json
from pipeline.tasks import get_task


def test_sql_task():
    task = get_task("sql")
    print("Task loaded successfully:", task.name)

    # 1. Test boxed extraction
    gen1 = "<think>Let me see.</think>\n<answer>\\boxed{SELECT name FROM head;}</answer>"
    extracted1 = task.extract_answer(gen1)
    print("Extracted 1:", extracted1)
    assert extracted1 == "SELECT name FROM head", f"Failed: {extracted1}"

    # 2. Test markdown inside boxed
    gen2 = "<think>Step by step.</think>\n<answer>\\boxed{```sql\nSELECT age FROM head;\n```}</answer>"
    extracted2 = task.extract_answer(gen2)
    print("Extracted 2:", extracted2)
    assert extracted2 == "SELECT age FROM head", f"Failed: {extracted2}"

    # 3. Test plain boxed without answer tags
    gen3 = "The final query is \\boxed{SELECT count(*) FROM singer};"
    extracted3 = task.extract_answer(gen3)
    print("Extracted 3:", extracted3)
    assert extracted3 == "SELECT count(*) FROM singer", f"Failed: {extracted3}"

    # 4. Test format_prompt
    primitive = {
        "problem": "Database Schema: ...\nQuestion: Find all singers.",
        "variant": "concert_singer",
        "level": "easy",
        "answer": "SELECT * FROM singer",
        "hints": ["Look at singer table", "Project all columns"],
    }
    prompt = task.format_prompt(
        primitive,
        "Problem:\n{problem}\n\n{hints_block}Please put your final SQL query in \\boxed{...}.",
    )
    print("Formatted prompt messages:", len(prompt))
    assert len(prompt) == 3
    assert "Here are some hints to help you:" in prompt[1]["content"]
    assert "Look at singer table" in prompt[1]["content"]

    # 5. Test check_correctness with real SQLite execution
    primitive_real = {
        "problem": "Schema...\nQuestion: Show stadium name and concert count.",
        "variant": "concert_singer",
        "level": "medium",
        "answer": "SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id",
        "metadata": {"source": "spider", "db_id": "concert_singer"},
    }

    # 5a. Equivalent query
    good_gen = "<think>I will join stadium and concert.</think>\n<answer>\\boxed{SELECT stadium.name, COUNT(concert.concert_id) FROM stadium INNER JOIN concert ON stadium.stadium_id = concert.stadium_id GROUP BY stadium.stadium_id}</answer>"
    is_correct, meta = task.check_correctness(primitive_real, good_gen)
    print("Correctness test (equivalent query):", is_correct, meta)
    assert is_correct is True

    # 5b. Wrong query
    bad_gen = "<think>I will select just stadium name.</think>\n<answer>\\boxed{SELECT name FROM stadium}</answer>"
    is_correct_bad, meta_bad = task.check_correctness(primitive_real, bad_gen)
    print("Correctness test (wrong query):", is_correct_bad, meta_bad)
    assert is_correct_bad is False

    # 5c. Syntax error query
    err_gen = "<think>Invalid syntax.</think>\n<answer>\\boxed{SELECT FROM WHERE bad}</answer>"
    is_correct_err, meta_err = task.check_correctness(primitive_real, err_gen)
    print("Correctness test (syntax error):", is_correct_err, meta_err)
    assert is_correct_err is False
    assert "pred_execution_error" in meta_err.get("error", "")

    # 6. Test metrics
    results = [
        {"level": "easy", "metadata": {"source": "spider"}, "correct": True},
        {"level": "easy", "metadata": {"source": "spider"}, "correct": False, "error": "pred_execution_error"},
        {"level": "hard", "metadata": {"source": "bird"}, "correct": True},
        {"level": "hard", "metadata": {"source": "bird"}, "correct": False, "error": "no_answer_tag"},
    ]
    metrics = task.compute_metrics(results)
    formatted = task.format_metrics(metrics, model_name="test-sql-model")
    print(formatted)

    print("\nALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_sql_task()
