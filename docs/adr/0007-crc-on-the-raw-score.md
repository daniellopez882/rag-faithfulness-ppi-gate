# ADR 0007: Choose the CRC threshold on the raw judge score

Status: accepted; amends the threshold scale in ADR 0003

## Context

ADR 0003 picks the flag threshold with conformal risk control "after calibration": `faithgate
calibrate` fitted the calibrator on its labelled rows and then ran CRC on that calibrator's
in-sample probabilities for the same rows. The CRC guarantee assumes the score was fixed before
the calibration labels were seen. A fitted calibrator is not:

* Platt scaling picks its slope, including its sign, from those labels. With an uninformative
  judge a negative slope reverses the order of the scores, and in a seeded simulation (30 rows,
  base rate 0.4, alpha 0.2) the expected slip risk came out near 0.22, above alpha.
* Histogram binning maps scores to per-bin label frequencies that need not be monotone, so the
  calibrated order is again a function of the labels.

For monotone non-decreasing calibrators (isotonic, temperature, Platt with a positive slope) the
calibrated rule flags a superset of what the same CRC rule on the raw score flags, so the bound
held there, but the code did not restrict itself to those.

## Decision

* `fit_artifact` runs CRC on the judge's raw score, which is a fixed function of the response.
  The calibrator still maps scores to probabilities for the PPI rate estimate and the report.
* The gate and the report flag a response when its raw score reaches the threshold, and show the
  raw score next to the calibrated probability.
* The artifact records `crc_scale = "raw score"`. Artifacts without it put the threshold on the
  calibrated scale; loading one fails with a message to refit rather than silently comparing a
  probability threshold with raw scores.
* The benchmark's CRC experiment runs on raw scores too, so its table describes what the CLI does.

## Consequences

* The flag set no longer depends on the calibration method; a test checks that every method
  yields the same threshold, and a second test checks the risk bound by simulation through
  `fit_artifact` with Platt and histogram calibrators.
* With a strictly increasing calibrator nothing changes. With isotonic regression the old rule
  flagged whole flat blocks; the raw rule flags slightly fewer responses at the same guaranteed
  risk.
* Existing `calibration.json` files have to be refitted once.

## Alternatives rejected

* Splitting the labelled rows, calibrator on one part and CRC on the other. Valid, but both fits
  lose rows, and the calibration sets in question are small.
* Mapping a raw-score threshold through the calibrator and flagging on probabilities. Equivalent
  only for monotone calibrators; it would mean constraining Platt's slope and dropping histogram
  binning for no gain over flagging on the raw score directly.
