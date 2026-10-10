# ADR 0006: A score-type upper bound, because budgets sit at low rates

Status: accepted; amends the interval construction in ADR 0001

## Context

ADR 0001 made the gate check a one-sided PPI++ upper bound built as estimate + z se (a Wald
bound). The coverage experiment behind it ran at RAGTruth's own hallucination rate, about 0.35,
where that bound came close to its level. A review pointed out that budgets are set much lower,
a few per cent, and two things go wrong there:

* With no hallucinated response among the labels, the label variance is 0, the plug-in lambda is
  0 and the standard error is 0, so the bound is exactly 0 and the gate passes any budget. At a
  true rate of 0.05 and 30 labels that happens in 0.95^30 = 21% of label draws.
* With one or two hallucinated labels the sample variance understates the variance at the rates
  the bound has to rule out, the same failure the Wald interval for a proportion has.

The README example uses a budget of 0.10 with the default `min_labelled = 30`, squarely in that
regime.

## Decision

* The gate and the report use a score-type interval by default (`kind="score"` in
  `faithgate.ppi.interval`). It inverts the normal approximation with the label variance taken at
  the hypothesised rate t, as the Wilson interval does, instead of at the sample mean.
* The judge term's variance reduction is estimated from the sample and carried over to t in two
  ways: the same absolute amount, and the same fraction of the label variance. The interval uses
  whichever variance is larger, and it never ends inside the Wald interval.
* The PPI++ lambda is clipped to [0, 1], between the classical estimate and PPI. From a few dozen
  labels with few positives the plug-in lambda is noisy, and values outside that range bought
  little and cost coverage in simulation.
* With lambda = 0 the bound is the Wilson bound with n - 1 in place of n, matching the ddof=1
  sample variance used elsewhere, so 30 clean labels give a 95% upper bound of
  z^2 / (29 + z^2) = 0.085, not 0.
* The Wald construction stays available as `kind="wald"`, and the benchmark reports both.

## Consequences

* The default test suite checks upper-bound coverage by seeded simulation at rates 0.02, 0.05
  and 0.10 with 30, 50 and 100 labels, and that the Wald bound fails one of those cells.
* The benchmark thins RAGTruth test to batches with rates of 0.05 and 0.10 and measures both
  bounds there; the README reports that table next to the one at RAGTruth's own rate.
* Taking the larger of the two variance forms is conservative when the judge is strong and the
  rate is moderate: the bound covers somewhat above its level there. That costs a little width,
  which the benchmark shows next to the Wald widths. For a gate, erring towards failing a build is
  the cheaper mistake.
* Artifacts and configs need no change; the bound only gets wider where it was too optimistic.

## Alternatives rejected

* Clopper-Pearson on the labels alone. Exact for the classical estimate, but it has no PPI
  version and would throw away the judge.
* A variance floor such as p(1 - p) with p = max(estimate, 1/n). Simple, but the floor is
  arbitrary, it ignores how much of the variance the judge explains, and it still uses the
  variance at the estimate rather than at the rate being tested.
* Refusing to report a bound when the labels hold no positive. That turns the most common
  outcome at low rates, a clean sample, into a failed build, and gives a team no way to pass
  with more labels short of finding a hallucination.
* The nonasymptotic betting (Waudby-Smith and Ramdas) bounds used in the PPI papers. Valid at any
  n, but with a few dozen labels and the error level split between the two terms they are much
  wider than a score bound that already holds its level in the measured range.
