This is a great framing problem, and I think there's a clean theoretical structure hiding in it. Let me lay out the framework I'd build the paper on, then turn each challenge into concrete measurements and experiments. Everything below is design — no results, no peeking at evals.

---

## 1. The central move: separate the solver from the controller

Treat every model in your zoo as two components sharing one set of weights:

- a **solver**: given a problem, produces an attempt;
- a **controller**: decides *what to do* — commit an answer, request a hint, abstain.

Then make two definitions that dissolve most of Challenge 1:

**Capability (of model $M$ on problem $x$).** $\kappa_M(x)$ = the probability that $M$ solves $x$ when *forced to answer* — estimated by sampling $k$ forced-answer generations (decoding constrained past the hint/abstain tokens, or prompted into answer-mode) at deployment temperature. Similarly $\kappa_M^{\text{hint}}(x)$ = forced-answer solve rate with a hint in context.

**Capability alignment.** $M$ is capability-aligned if its controller is near-optimal *for the task reward, given $M$'s own $\kappa$ and $\kappa^{\text{hint}}$*. Formally, alignment is low **decision regret**: the gap between the expected reward of $M$'s actual decisions and the expected reward of the best possible controller wired to $M$'s own solver.

Notice what this buys you:

1. **Capability is self-referential and counterfactual, so there is no circularity.** You never define $M$'s capability by its behavior (which the training shaped) — you define it by what happens when the decision is taken away. It's measurable for any checkpoint, including RLVR models and base models, with no proxy model anywhere in the definition.
2. **The RLVR sibling is demoted from "ground truth for capability" to a matched-compute control.** It no longer needs to be justified as *defining* what the hint model can do. It answers a different, separable question: *did the hint mechanism cost competence?* (Compare $\kappa_{\text{hint}}$ vs $\kappa_{\text{RLVR}}$ trained from the same SFT parent with matched steps/data.) That's a capability-*preservation* claim, cleanly separated from the capability-*alignment* claim, which is entirely internal to one model.

One refinement worth a subsection in the paper, because "capability" is genuinely ambiguous: distinguish (i) **attempt-level capability** ($\kappa$ as above — the success probability of the very attempt the model would make), (ii) **elicitation ceiling** (pass@$k$ / best-of-$n$), and (iii) **learnable frontier** (what it could solve after more training). The alignment claim only needs (i), which is the decision-relevant one: when the controller chooses "answer," what matters is whether *that attempt* will succeed. (ii) shows up in the capability-preservation analysis, (iii) in the abstention-debt analysis below. Saying this explicitly preempts the reviewer who says "capability is ill-defined."

---

## 2. The theory section: reward structures as elicitation mechanisms

Here's the theoretical backbone that connects "hint-seeking" to "capability alignment," and it's provable in a two-line decision model.

Let $a$ be the abstention reward, $c$ a per-hint price, $p = \kappa(x)$, $q = \kappa^{\text{hint}}(x)$.

- **RLVR** (correct $= 1$, else $0$): always-answering is optimal *regardless of $p$*. The objective's optimum is invariant to the model's self-knowledge — metacognition is **unidentified**. This is the formal version of "best guess": RLVR doesn't just fail to teach self-assessment, it makes self-assessment economically irrelevant. (This connects directly to the argument that binary grading incentivizes guessing — the OpenAI 2025 hallucination paper is your antagonist citation.)
- **Abstention at reward $a$**: optimal controller answers iff $p \ge a$. The abstain reward *is* a confidence threshold; behavior reveals one bit about $p$ per problem. (Your abstention reward of $a = 0.5$ literally announces "answer iff you're better than a coin flip.")
- **Hint at price $c$**: optimal controller compares $p$, $a$, and $\max(q, a) - c$. It answers when $p$ is high, buys a hint when the *lift* $q - p$ exceeds $c$ and $q - c$ beats abstaining, and abstains directly when both $p$ and $q$ are low. Behavior now reveals joint information about $(p, q)$ — this is a value-of-information decision, strictly richer than abstention.
- **Price-conditioned** (announce $a$ or $c$ in the prompt, sampled per episode, reward computed with the announced price): sweeping the price at inference traces out the model's *entire* confidence curve, the way BDM mechanisms elicit willingness-to-pay in economics. Willingness to pay for help = revealed confidence.

That's an **identification hierarchy**: RLVR identifies nothing about the model's self-assessment, abstention identifies a threshold, hint-seeking identifies value-of-information structure, price-conditioning identifies the full curve. Each reward design is a progressively stronger *proper scoring rule over decisions*. This is, I think, the paper's actual thesis:

> Help-seeking reward structures make honest self-assessment economically necessary; correctness-only RLVR makes it economically irrelevant. We measure how fully trained models realize the self-knowledge their objective demands.

Two obligations this creates:

- **Derive the optimal controller for your *implemented* reward** (the hint-bonus-for-wrong-answers form, multi-turn, with whatever turn caps apply — a small dynamic program). You need this anyway as the normative overlay in every figure, and it will expose any degenerate optimum (e.g., insurance-farming on the wrong-answer bonus) *before* you burn more training runs. Note that a bonus-on-wrong-with-hint form and a per-hint-cost form imply different thresholds; the derivation tells you what your models were actually incentivized to reveal.
- **Be precise about who initiates the hint.** If the environment auto-serves a hint after a wrong answer, the model's revealed decision is *commit-vs-continue*, not *seek-vs-answer*. The framework covers both (any decision with informational consequences works), but the cleanest "the model knows when it needs help" story wants model-initiated requests with an explicit price. If your current multi-turn setup is environment-initiated, say so and frame the identified quantity honestly — or add the model-initiated variant as the flagship run.

This also retroactively organizes your method zoo: `simple` = no controller, `verify` = buys evidence about its own attempt, `abstention_verify` = threshold controller, `hint_encourage` = information-purchase controller. Your four results rows become a designed comparison along one axis — *what decisions does the objective let the model express, and how much metacognition does each identify* — rather than four methods that happen to coexist.

---

## 3. Metrics: what "demonstrating alignment" means operationally (Challenge 2)

For a probe set of problems, estimate $\hat{\kappa}$ and $\hat{\kappa}^{\text{hint}}$ per problem ($k \approx 16$–$32$ forced samples), and record natural-behavior decisions. Then:

1. **Behavioral calibration curve** (the paper's Figure 1). Bin problems by the model's *own* $\hat{\kappa}$; plot $P(\text{seek/abstain} \mid \text{bin})$; overlay the reward-optimal step function from the derivation. A capability-aligned model shows a sharp sigmoid at the derived threshold; RLVR is a flat line at zero. The caption stat everyone will quote: "answers problems it would solve at 78%, declines problems it would solve at 19%."
2. **Metacognitive efficiency (headline number).**
   $$\mathrm{ME} = \frac{V_\pi - V_{\text{const}}}{V_{\text{oracle}} - V_{\text{const}}}$$
   where $V_\pi$ is the achieved decision-layer reward, $V_{\text{oracle}}$ is the best controller applied to the model's own $(\hat{\kappa}, \hat{\kappa}^{\text{hint}})$, and $V_{\text{const}}$ is the best *constant* policy (always-answer / always-seek / always-abstain). $\mathrm{ME}$ measures the fraction of the value of self-knowledge that the policy actually realizes. Constant policies — including RLVR — sit at $\le 0$ by construction; the oracle is $1$. It penalizes both failure modes symmetrically: overconfidence (the RLVR limit) and learned helplessness / hint-addiction (the over-seeking limit), and the calibration curve shows which side you're on. Cross-fit the oracle (fit thresholds on one half of samples, evaluate on the other) or it inflates on estimation noise.
3. **Decision informativeness.** Mutual information between the visible decision and success. For any constant policy this is exactly zero, which makes the claim crisp: hint training raises $I(\text{decision};\, \text{success})$ from $0$ to $X$ bits.
4. **Self- vs difficulty-alignment (the answer to "its OWN capabilities").** An abstainer that merely learned "geometry is hard" is *difficulty-aligned*, not capability-aligned. Test: does $M$'s decision predict $M$'s own $\hat{\kappa}$ better than it predicts another model's $\hat{\kappa}$ or the cross-model consensus difficulty? Report $\Delta\text{AUROC}$, plus a regression of decisions on own-$\hat{\kappa}$ controlling for consensus-$\hat{\kappa}$ and surface features. This distinction — difficulty-aligned vs self-aligned — is worth naming; it's the paper's sharpest conceptual contribution.
5. **Risk–coverage curves at matched coverage.** Selective accuracy vs fraction answered, against the baselines below. Never report raw accuracy-on-answered anywhere in the paper — higher selective accuracy is trivially purchasable with lower coverage, and any comparison not at matched coverage will (correctly) get the paper killed.

---

## 4. The experiments, prioritized

**E1 — Core alignment measurement.** Forced-answer harness + probe set (~500 problems per task spanning difficulty), compute metrics 1–3 for all four method rows $\times$ three sizes. This is the paper's spine and runs on checkpoints you already have.

**E2 — The baseline gauntlet (the paper lives or dies here).** The obvious rebuttal is: "why train any of this in? Take the RLVR model and bolt on post-hoc selection." So the hint model must beat, at matched coverage: (a) RLVR + self-consistency vote-share thresholding, (b) RLVR + verbalized confidence thresholding, (c) RLVR + $P(\text{True})$-style self-evaluation, (d) prompted abstention ("say IDK if unsure"), (e) the SFT-only abstention parent (is RL even needed?). Two honest outcomes: the hint model's single trajectory matches $k$-sample vote-share AUROC at $1/k$ the inference cost — an *amortized self-assessment* claim, which is a fine win — or it strictly dominates the curve. Also report the token-cost axis explicitly; it favors you.

**E3 — Cross-model disagreement (the "OWN" clincher, axis 1).** You have three sizes per task. Select problems where $\hat{\kappa}_{1.5\text{B}}$ and $\hat{\kappa}_{4\text{B}}$ disagree strongly (select on one sample-split, evaluate on the other, or regression-to-the-mean will fabricate your result). Capability alignment predicts each model's decisions follow *its own* $\kappa$ on the disagreement set — the small model seeks where the large one commits and vice versa. Difficulty-alignment predicts they agree everywhere.

**E4 — Certified-impossible probes (estimation-free ground truth).** Countdown lets you construct instances that provably have no solution (exhaustive check), and code-output can use unseeded randomness — *aleatoric* impossibility, $\kappa = 0$ for every model by construction, no sampling needed. Predictions: RLVR answers confidently ~100% of the time; the abstention model abstains; and — the beautiful one — a truly VOI-rational hint model with an abstain option should **abstain without purchasing a hint**, because the hint lift is provably zero. That prediction separates "seeks when unsure" (a heuristic) from genuine value-of-information reasoning. Distinguish aleatoric from epistemic difficulty explicitly in the taxonomy.

**E5 — OOD dose-response.** You already have 2op/7op/8op countdown splits. As operator count grows past training, $\kappa$ collapses; plot seek/abstain rate against $1 - \hat{\kappa}$ across the sweep. Item-level memorization is impossible on unseen items, so tracking here is generalization of the *mechanism*.

**E6 — In-context capability interventions (axis 2: causal, within-item).** Change the capability without changing the problem: prepend a worked example of the same skill (verify it raises $\hat{\kappa}$ by forced sampling), and a matched-length irrelevant example as placebo. An aligned model's seek-rate should drop in proportion to $\Delta\hat{\kappa}$ under the real exemplar and not move under the placebo or under content-free "encouragement" text. Cheap, no training, and it upgrades the whole story from correlational to causal.

**E7 — Temporal tracking across checkpoints (axis 3: within-run).** You save every 25 steps. For a fixed probe set, track $\hat{\kappa}_t$ and the decision through training. Prediction: problems the model learns during RL flip from seek $\to$ answer at roughly the step where $\hat{\kappa}_t$ crosses the derived threshold (event-study plot aligned at crossing time; control for global drift using never-crossing problems). Same data answers the **abstention-debt question**, which you should confront rather than hide: on the set the RLVR sibling *learned* during RL, did the hint model also learn (possibly scaffolded by hinted rollouts — check whether unhinted $\hat{\kappa}$ rises on frequently-hinted problems), or did early abstention starve it of gradient and entrench incapacity? Either answer is a real finding; a feedback loop where abstention becomes self-fulfilling is exactly what a critical reviewer will suspect, so measure it first.

**E8 — Price-conditioned model (one new run, the rationality test).** Sample the hint price per episode, announce it in the prompt, reward accordingly. At eval, sweep the announced price: do behavioral thresholds shift quantitatively as the derivation predicts? A fixed-price model could in principle learn a static difficulty classifier; a model whose threshold tracks announced prices must represent something like a continuous internal confidence. Bonus: this gives you a *deployment knob* (set the price to the true cost of human/tool assistance and the model makes economically optimal escalation decisions) and turns your risk–coverage point into a full curve.

**E9 (optional, mechanistic garnish).** Kadavath-style linear probe for $P(\text{I know})$ on hidden states at the decision token, across base/SFT/RLVR/hint checkpoints. The candidate narrative — RLVR models still *represent* their competence but are behaviorally disconnected from it, while hint-RL wires the representation to action ("closing the knowledge–behavior gap") — would be a memorable result if it holds.

Triangulation summary for Challenge 2: self-vs-difficulty is separated on three independent axes — across models (E3), across time (E7), and causally within-item (E6). Any one is arguable; three concordant axes are hard to dismiss.

---

## 5. Threats to validity to design against now

- **Forced-mode validity.** Forcing answers might be off-distribution for an abstention-trained model, biasing $\hat{\kappa}$ down exactly on abstained items. Validate: (a) on problems the model answers naturally, forced accuracy must match natural accuracy; (b) compare token-mask forcing vs prompt-variant forcing; (c) sanity-bound against the matched-compute sibling — if forced $\hat{\kappa} \approx 0$ where the RLVR sibling scores 60%, be suspicious.
- **Estimation noise in $\hat{\kappa}$.** Binomial noise attenuates every correlation and AUROC. Fix $k$ with a power calculation, report split-half reliability, disattenuate, and always select subsets on one split and evaluate on the other.
- **Hint leakage / degenerate optima.** If hints leak answers, always-seek dominates and behavior measures reward misdesign, not metacognition. Measure the lift distribution $\hat{q} - \hat{\kappa}$ and confirm the price sits inside the lift band. The reward derivation (Section 2) is the pre-flight check.
- **Coverage confound.** Matched coverage everywhere, no exceptions.
- **Pre-registration.** Derive thresholds and freeze the analysis before touching final evals — which conveniently matches the discipline you're already imposing on this conversation.
- **Naming.** "Capability alignment" collides with safety-alignment vocabulary; consider "capability-calibrated decision-making" or "metacognitive alignment," or keep the term with an explicit disclaimer footnote.

---

## 6. Paper skeleton and positioning

Abstract logic: (1) correctness-only RLVR leaves the answer/seek/abstain decision unidentified — guessing is optimal regardless of self-knowledge; (2) priced help-seeking makes capability-aligned decisions the unique optimum (identification hierarchy); (3) empirically, hint-trained models approach that optimum: decisions track their *own* forced-answer solve rates rather than item difficulty (E3/E6/E7), respect value-of-information structure including abstaining without seeking on provably unsolvable items (E4), generalize OOD (E5), and shift thresholds with announced prices (E8) — at $\mathrm{ME} = X$ vs $\le 0$ for RLVR, without competence loss vs a matched-compute RLVR control.

Related-work anchors (from memory — verify with a fresh search, and specifically look for 2025 work on RLVR degrading refusal/abstention, which I believe exists, e.g. a "hallucination tax of reinforcement finetuning" paper, plus any concurrent RL-with-IDK work): selective prediction (Chow 1970; El-Yaniv & Wiener; Geifman & El-Yaniv), learning-to-defer (Madras et al.; Mozannar & Sontag), action advising in RL (teacher–student advice budgets), proper scoring rules (Gneiting & Raftery) and BDM elicitation, "LMs (Mostly) Know What They Know" (Kadavath et al.), verbalized uncertainty (Lin et al.), semantic entropy (Farquhar et al.), R-Tuning / honesty alignment, Kalai–Vempala on calibration and hallucination, OpenAI 2025 on binary grading incentivizing guessing, RLHF calibration degradation (GPT-4 report), and IRT-based LLM evaluation for the difficulty/ability decomposition.

Also worth one paragraph in the intro: why this is not "just calibration." Stated probabilities are cheap talk — trainable and ignorable at decision time; a model can verbalize 30% confidence and still guess. Capability alignment is *revealed* confidence: incentive-compatible because decisions carry consequences, and it composes directly with downstream reward, which is what matters for agents deciding when to escalate to a human or a tool.

---

## Minimal decisive path

If you can only do three things: the reward derivation (Section 2, a weekend of math, and it de-risks everything), then **E1 + E2 + E3** on checkpoints you already have. That's a defensible paper. E4 and E6 are cheap and upgrade it from correlational to causal; E8 is the one new training run I'd fight for, because price-responsiveness is the single strongest evidence that the model carries a continuous internal confidence rather than a memorized difficulty classifier.

If useful, I can also turn this into a polished one-page research-design artifact for sharing with collaborators — just say the word.
