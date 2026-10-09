# ADR 0002: A 13M-parameter binary NLI model with SummaC over word-budget chunks

Status: accepted

## Context

The faithfulness judge has to run inside CI on a CPU, over thousands of responses whose sources
reach 2,000 tokens (RAGTruth summaries and Yelp records). SummaC scores a response by running NLI
between every response sentence and every source piece, so cost is (units x chunks) forward
passes per response.

A quick probe on the development machine (i5-8500, 6 cores, onnxruntime; a one-off measurement,
not part of the recorded benchmark) put the pinned `xtremedistil-l6-h256-zeroshot-v1.1-all-33`
model at about 10 ms per pair at 128 tokens, 20 ms at 256 and 90 ms at 512. The int8 export in
the same repository was no faster on this CPU (no VNNI) and moved the entailment probability by as
much as 0.55 on eight probe pairs, so only the fp32 export is used.

Options considered:

1. A 400M-parameter DeBERTa NLI model. Roughly 30x the compute per pair; scoring RAGTruth test
   would take many hours on this CPU, which rules it out for a CI gate.
2. Sentence-level premises, as in SummaC's best configuration. Most accurate per pair for models
   trained on short premises, but the pair count grows with the number of source sentences.
3. Word-budget chunks of consecutive source sentences with a one-sentence overlap.
4. An LLM judge. Kept behind the provider interface; not run here, so it has no numbers.

## Decision

Use the small binary model through onnxruntime and tokenizers (no torch at runtime), score
sentence or claim units against word-budget chunks, and choose the budget and unit type on a dev
slice of RAGTruth train (never on test). With a binary entailment head the SummaC term
P(entail) - P(contradict) becomes 2 P(entail) - 1, so the zero-shot row maximum is taken on
P(entail) directly. Both SummaC aggregations are kept: the zero-shot mean (and a min variant that
lets one unsupported unit decide), and SummaC-Conv trained on a calibration slice of train, with
mean or log-sum-exp pooling.

Yelp records are linearised into short sentences ("X is located at ...", 12-hour opening times,
one sentence per attribute) and null attributes are dropped, because the small model cannot read
nested JSON.

## Consequences

* The judge fits a CI budget, and the benchmark reports its throughput on the target CPU.
* Accuracy is limited by the model: multi-fact sentences whose evidence spans two chunks are not
  entailed by any single chunk and look unsupported. The benchmark compares against a unigram
  overlap baseline so that this limit is visible rather than hidden.
* The chunk budget is a tuned hyperparameter recorded in the results JSON with the dev scores of
  every candidate.
