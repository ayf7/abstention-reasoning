"""Competition Math v2.

Same problems, hints and correctness checking as competition_math. v2 changes
two things about how the model asks for a hint, both aimed at making the request
a real decision rather than a memorised move.

First, the lead-in is gone. v1 spliced one of seven canned "I'm stuck..."
phrases in front of every forced <request></request>, so the SFT model learned
the phrase rather than the decision and reproduced it verbatim in 85% of its RL
rollouts. v2 cuts the reasoning at a natural boundary and requests with no
lead-in (see the hint_transition flag on Method).

Second, the request lives *inside* the think block instead of after it, so
</think> is reached exactly once, immediately before <answer> (the
nested_request flag). This is what makes the counterfactual eval possible:
`evaluate --no-hints` bans the request token, and the model is left still
reasoning and free to continue, rather than stranded past its own </think> with
nothing to do but answer cold.

It is a separate task rather than a second method on competition_math because
the two share a method name (hint) but must not share primitives, prompts,
datasets or model directories.
"""

import importlib.util
from pathlib import Path

from pipeline.tasks.competition_math.task import CompetitionMathTask

# The nested tag grammar is defined once, in the reward function verl scores
# rollouts with. Loading it here keeps SFT filtering and RL scoring from drifting
# apart; it is loaded by path because the recipe is a standalone file, not a
# package (which is also how the recipe itself pulls in v1).
_REWARD_PATH = (Path(__file__).resolve().parents[3]
                / "verl" / "recipe" / "competition_math_v2" / "reward_function.py")
_spec = importlib.util.spec_from_file_location("competition_math_v2_reward", _REWARD_PATH)
_reward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reward)
_has_malformed_structure = _reward.has_malformed_structure


class CompetitionMathV2Task(CompetitionMathTask):
    name = "competition_math_v2"

    def filter_for_sft(
        self,
        examples: list[dict],
        include_abstained: bool = True,
        include_wrong_valid_format: bool = False,
    ) -> list[dict]:
        """v1's filter, plus a structural check on the nested grammar.

        v1 keeps anything marked correct. In the full 14B run that let through 20
        generations that were correct but structurally invalid -- 4 of them closed
        </think> before requesting a hint, the v1 habit. Training on those teaches
        back the very pattern the nested format exists to remove, and it only takes
        a few examples for the model to learn that </think> is a legal place to
        stop and ask.
        """
        kept = super().filter_for_sft(
            examples,
            include_abstained=include_abstained,
            include_wrong_valid_format=include_wrong_valid_format,
        )
        return [ex for ex in kept
                if not _has_malformed_structure(ex.get("generation", ""))]
