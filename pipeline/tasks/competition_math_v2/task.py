"""Competition Math v2.

Same problems, hints, prompts and correctness checking as competition_math. The
only thing v2 changes is how SFT hint data is generated: v1 spliced one of seven
canned "I'm stuck..." phrases in front of every forced <request></request>, so
the SFT model learned the phrase rather than the decision, and reproduced it
verbatim in 85% of its RL rollouts. v2 cuts the reasoning at a natural boundary
and requests with no lead-in, which makes <request></request> an actual decision
point (see the hint_transition flag on Method).

It is a separate task rather than a second method on competition_math because
the two share a method name (hint) but must not share primitives, prompts,
datasets or model directories.
"""

from pipeline.tasks.competition_math.task import CompetitionMathTask


class CompetitionMathV2Task(CompetitionMathTask):
    name = "competition_math_v2"
