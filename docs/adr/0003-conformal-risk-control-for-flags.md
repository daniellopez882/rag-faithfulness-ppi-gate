# ADR 0003: Conformal risk control for the per-response flag threshold

Status: accepted

## Context

Besides the batch-level rate, the gate lists individual responses for a human to look at. That
needs a threshold on the judge's score. A threshold of 0.5 on an uncalibrated score has no meaning,
and the F1-optimal threshold on a labelled set optimises a trade-off nobody asked for and carries
no statement about what slips through.

What a reviewer wants to know is how much hallucinated output the gate lets past without a flag.

## Decision

Pick the flag threshold with conformal risk control (Angelopoulos et al., 2022) on a labelled
calibration set, after calibration:

    t_hat = sup { t : (n * R_n(t) + 1) / (n + 1) <= alpha }

with one of two monotone losses:

* `slip` (default): 1 if a response is hallucinated and not flagged. Guarantees that the expected
  share of all responses that are hallucinated and unflagged is at most alpha.
* `miss`: the same loss over hallucinated responses only, which bounds the expected miss rate
  among hallucinated responses.

The threshold, alpha, the loss and the calibration size are stored in `calibration.json` with the
calibrator, so the gate applies exactly what was fitted.

## Consequences

* The guarantee holds in expectation over calibration sets, for responses exchangeable with the
  calibration set. A new generator model or prompt template breaks exchangeability; the threshold
  has to be refitted on labelled outputs of the system being gated.
* With a weak judge the guarantee is met by flagging a lot: the flag rate is the price, and the
  benchmark reports it next to the realised risk.
* If alpha is below 1/(n + 1) the only safe threshold flags everything; the code returns
  `-inf` rather than pretending.
* Alternatives rejected: a fixed threshold (no guarantee), split-conformal prediction sets for the
  binary label (guarantees coverage of the label set, which does not translate into a bound on
  missed hallucinations), and Neyman-Pearson style tuning without the finite-sample correction.
