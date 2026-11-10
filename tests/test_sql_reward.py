"""Tests for verl/recipe/sql/reward_function.py."""

import importlib.util
from pathlib import Path


def test_reward_function():
    reward_path = Path("verl/recipe/sql/reward_function.py")
    spec = importlib.util.spec_from_file_location("sql_reward", reward_path)
    reward_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_mod)

    # 1. Test answer extraction
    sol1 = "<think>Some thought.</think>\n\n<answer>\\boxed{SELECT * FROM singer}</answer>"
    ans1 = reward_mod.extract_answer(sol1)
    assert ans1 == "SELECT * FROM singer", f"Failed: {ans1}"

    # 2. Test extraction with markdown inside boxed
    sol2 = "<think>Thought.</think>\n\n<answer>\\boxed{```sql\nSELECT count(*) FROM concert\n```}</answer>"
    ans2 = reward_mod.extract_answer(sol2)
    assert ans2 == "SELECT count(*) FROM concert", f"Failed: {ans2}"

    # 3. Test correct query scoring
    ground_truth = {
        "db_id": "concert_singer",
        "answer": "SELECT T2.name , count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id GROUP BY T1.stadium_id",
    }
    sol_correct = (
        "Let me write the query.</think>\n\n"
        "<answer>\\boxed{SELECT stadium.name, COUNT(concert.concert_id) FROM stadium "
        "JOIN concert ON stadium.stadium_id = concert.stadium_id GROUP BY stadium.stadium_id}</answer>"
    )
    res_correct = reward_mod.compute_score(
        data_source="sql",
        solution_str=sol_correct,
        ground_truth=ground_truth,
        extra_info={},
    )
    print("Correct query score result:", res_correct)
    assert res_correct["correct"] is True
    assert res_correct["score"] == 1.0

    # 4. Test wrong query scoring (well formatted)
    sol_wrong = (
        "Wrong thought.</think>\n\n"
        "<answer>\\boxed{SELECT name FROM singer}</answer>"
    )
    res_wrong = reward_mod.compute_score(
        data_source="sql",
        solution_str=sol_wrong,
        ground_truth=ground_truth,
        extra_info={},
        format_score=0.1,
    )
    print("Wrong query score result:", res_wrong)
    assert res_wrong["correct"] is False
    assert res_wrong["score"] == 0.1

    # 5. Test hint penalty scoring
    sol_hint = (
        "I am stuck.</think>"
        "<request></request><response>Check stadium table</response><think>Now I know.</think>\n\n"
        "<answer>\\boxed{SELECT stadium.name, COUNT(concert.concert_id) FROM stadium "
        "JOIN concert ON stadium.stadium_id = concert.stadium_id GROUP BY stadium.stadium_id}</answer>"
    )
    res_hint = reward_mod.compute_score_hint(
        data_source="sql",
        solution_str=sol_hint,
        ground_truth=ground_truth,
        extra_info={},
        hint_penalty=0.2,
    )
    print("Hint penalty score result:", res_hint)
    assert res_hint["correct"] is True
    assert res_hint["num_hints"] == 1
    assert abs(res_hint["score"] - 0.8) < 1e-4

    # 6. Test malformed tag structure
    sol_malformed = (
        "<think>Incomplete tags"
    )
    res_malformed = reward_mod.compute_score(
        data_source="sql",
        solution_str=sol_malformed,
        ground_truth=ground_truth,
        extra_info={},
    )
    print("Malformed score result:", res_malformed)
    assert res_malformed["malformed"] is True
    assert res_malformed["score"] == 0.0

    print("\nALL REWARD FUNCTION TESTS PASSED!")


if __name__ == "__main__":
    test_reward_function()
