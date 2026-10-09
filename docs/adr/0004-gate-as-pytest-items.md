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
`pytest.File` node for the config: `rag_gate_rate` for the rate bound and, with
`per_response = true`, one `rag_gate_response[<id>]` item per response. Failures raise a dedicated
exception rendered as a one-line reason without a traceback. A terminal summary section prints the
estimate, the bound, lambda and the flag count. Configuration errors become pytest usage errors.

## Consequences

* Three lines in a workflow are enough: score, (calibrate once elsewhere), `pytest --rag-gate`.
* The plugin is loaded through the `pytest11` entry point, so installing the package activates it;
  without the option it adds nothing.
* Scoring is not done inside pytest: the gate reads scores written by `faithgate score`. This keeps
  model downloads and minutes of NLI out of the test process and lets a cached score file be gated
  repeatedly.
* The plugin is tested with `pytester`, including selection with `-k` and the usage-error path.
