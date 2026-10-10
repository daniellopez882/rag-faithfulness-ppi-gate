# ADR 0004: The gate is a pytest plugin that adds items from a TOML file

Status: accepted

## Context

Teams already run pytest in CI and already know how to read its output, select tests with `-k`
and mark expected failures. A separate exit-code CLI would need its own reporting and its own
wiring in every pipeline.

Ways to put the gate into pytest:

1. A collector for `*.toml` files found during collection. Only works when the file sits inside a
   collected directory, which ties the config's location to the test layout.
2. A fixture users call from their own test function. Every team writes the same boilerplate and
   failure messages vary.
3. A command-line option that names the config and adds items to the session.

## Decision

`--rag-gate path/to/rag-gate.toml` evaluates the gate once during collection
(`pytest_collection_modifyitems`, tryfirst so `-k` and `-m` still apply) and appends items under a
`pytest.File` node for the config. `rag_gate_rate` checks the rate bound. With
`per_response = true` there is also one `rag_gate_response[<id>]` item per response, failing at
the CRC threshold; that is off by default (see below). Failures raise a dedicated exception
rendered as a one-line reason without a traceback. A terminal summary section prints the
estimate, the bound, lambda and how many responses reach the flag threshold. Configuration errors,
including a `calibration.json` fitted for a different judge than the scores, become pytest usage
errors.

Two defaults were changed after the RAGTruth benchmark:

* `per_response = false`. With the judges measured on RAGTruth, the CRC threshold at the
  `calibrate` default (slip risk, alpha 0.10) flags about 70% of test responses, so one failing
  item per flag would fail almost every build. The flags are a review
  queue (ADR 0003); `faithgate report` lists them.
* `too_few_labels = "fail"`. Below `min_labelled` labels there is no bound to check. Skipping the
  item would leave the build green with nothing checked, so the item fails unless the config opts
  into `"skip"`.

## Consequences

* Two workflow lines run the gate: `faithgate score` and `pytest --rag-gate`; calibration is a
  one-off step on labelled outputs, outside the pipeline.
* The plugin is loaded through the `pytest11` entry point, so installing the package activates it;
  without the option it adds nothing.
* Scoring is not done inside pytest: the gate reads scores written by `faithgate score`. This keeps
  model downloads and minutes of NLI out of the test process and lets a cached score file be gated
  repeatedly.
* The plugin is tested with `pytester`, including selection with `-k` and the usage-error path.
