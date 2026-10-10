# ADR 0005: Keep the dev rule for the primary judge, and report a length baseline next to it

Status: accepted

## Context

The benchmark picks one NLI judge as "primary": its calibrator drives the PPI, CRC and span
experiments, and `faithgate score` defaults to it. The rule, fixed before test was scored, is the
highest AUROC on a 150-response dev slice of RAGTruth train.

A review of the first full run showed that the judge this rule picked, SummaC-Conv with log-sum-exp
pooling, ranked responses mostly by length. Log-sum-exp over n unit logits adds up to log n to
the response logit, and RAGTruth's hallucination rate rises with response length, so a longer
response scores higher whatever the NLI model says. The README gave the judge's AUROC with no
length comparison.

The same review found the sentence splitter merging sentences after words such as "no." and
acronyms such as "U.S.". After that fix the dev rule picked `nli-zs-mean` instead, by 0.0016 dev
AUROC.

## Decision

* Keep the dev-AUROC rule. Switching to a rule chosen after seeing test results would be
  selection on test.
* Add a model-free baseline, the number of checkable units in the response, to every judge table.
* Report a length-matched AUROC for every judge: AUROC over only the hallucinated/faithful pairs
  that share a task type and a unit count (`faithgate.metrics.matched_auroc`). A score that only
  tracks length gets exactly 0.5.
* Report the dev length-matched AUROC too, without using it to choose.
* Keep log-sum-exp as a Conv pooling option. For "any span hallucinated" labels it behaves like a
  noisy-OR over units, where a longer response really does have more chances to contain one
  unsupported claim. The length-matched AUROC shows how much the judge adds beyond that.

Later the README renamed the length-matched AUROC to "task- and length-matched", because its
strata hold task type fixed as well, and the benchmark added a task-matched AUROC so the drop
from the pooled figure can be split between task-type base rates and length. The JSON keys are
now `dev_auroc_task_length_matched` and `meta_eval.*.auroc_task_length_matched`.

## Consequences

* The README states plainly how much of each judge's pooled AUROC is length. On test the
  primary judge keeps a length-matched AUROC above 0.5, while `nli-conv-max` drops close to it.
* The dev slice is too small to choose on length-matched AUROC: on dev that criterion prefers
  `nli-conv-mean`, which is the weaker judge on test by the same measure. The numbers are in
  `bench/results/full.json` (`dev_auroc_length_matched`, `meta_eval.*.auroc_length_matched`).
* A primary choice that a splitter fix can flip is a fragile choice. The README says so in its
  limitations rather than presenting the primary judge as clearly best.

## Alternatives rejected

* Log-mean-exp pooling (log-sum-exp minus log n). It removes the length term by construction, and
  with it a real effect. It would also have meant a new judge chosen after seeing test.
* Choosing the primary judge on test. Every test number would then be optimistic.
* Dropping the primary judge and reporting only the overlap baseline. The PPI and CRC experiments
  are meant to show how the statistics behave with a weak model-based judge, which is the common
  case.
