# ADR 0001: PPI++ upper bound as the gate's rate check

Status: accepted

## Context

The gate has to answer one question per release: is the hallucination rate of this batch of RAG
outputs below a budget? Two sources of evidence exist. A judge (NLI, overlap or an LLM) can score
every response but is biased: on RAGTruth the raw NLI score is miscalibrated and its error differs
by task type. Human labels are unbiased but expensive; a team can afford tens to a few hundred per
release, not thousands.

Options considered:

1. Classical interval on the human labels alone. Valid, but its width shrinks only with the number
   of labels, so a 100-label budget gives an interval roughly 0.18 wide at a 30 % rate.
2. The judge's mean over the whole batch. Narrow, and wrong by exactly the judge's bias. No
   amount of data fixes it, and the bias is not stable across task types or generator models.
3. Prediction-powered inference (PPI): judge mean on the unlabelled responses plus a correction
   measured on the labelled ones. Valid for any judge; width depends on how well the judge tracks
   the labels.
4. PPI++: PPI with the judge term scaled by a data-driven lambda. lambda = 0 is the classical
   estimator, lambda = 1 is PPI; the plug-in lambda minimises the asymptotic variance.
5. Bootstrap or Bayesian alternatives over the same data. More machinery, no stronger guarantee
   for a mean, and slower inside a test run.

## Decision

The gate computes a one-sided PPI++ upper confidence bound with the plug-in lambda
`Cov(Y, f) / ((1 + n/N) Var(f))`, unclipped, and fails when the bound exceeds `max_rate`. The
judge input `f` is the calibrated probability from `faithgate calibrate`, fitted on labelled data
that is not reused for the rate check. Classical and plain PPI stay available through
`estimator = ...` and are always shown in the report next to the judge-only estimate.

## Consequences

* Validity rests on the labelled responses being a uniform random subset of the same batch. The
  gate cannot check that; the README and the report say so. Labelling only the responses a
  reviewer found suspicious breaks the guarantee.
* The intervals are CLT intervals, valid asymptotically. The gate refuses to compute a bound below
  `min_labelled` labels (30 by default) and the benchmark measures empirical coverage at n = 50 to
  500 on RAGTruth to show how close to nominal it gets.
* A useless judge costs little: lambda goes towards 0 and the bound approaches the classical one.
  A good judge narrows the bound, which is the only lever that lets fewer labels pass a tight
  budget.
* Calibration matters for width, not validity: PPI is valid with the raw score too, which the
  benchmark checks by running the coverage experiment on raw and calibrated scores.
