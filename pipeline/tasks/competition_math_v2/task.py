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

from pipeline.tasks.competition_math.task import CompetitionMathTask


class CompetitionMathV2Task(CompetitionMathTask):
    name = "competition_math_v2"
