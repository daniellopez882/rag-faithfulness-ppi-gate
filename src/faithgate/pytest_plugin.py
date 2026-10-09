"""pytest plugin: ``pytest --rag-gate rag-gate.toml`` adds gate items to the session.

* ``rag_gate_rate`` fails when the one-sided upper confidence bound on the hallucination rate
  exceeds ``max_rate`` (skipped when there are fewer than ``min_labelled`` human labels).
* ``rag_gate_response[<id>]`` (with ``per_response = true``) fails for each response whose
  calibrated probability reaches the conformal-risk-control threshold.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from faithgate.gate import GateOutcome, RateCheck, ResponseCheck, evaluate, load_config

_OUTCOME = pytest.StashKey[GateOutcome]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rag-gate", "RAG faithfulness gate")
    group.addoption(
        "--rag-gate",
        dest="rag_gate",
        metavar="TOML",
        default=None,
        help="gate config with a [rag_gate] table; adds rate and per-response gate items",
    )


class GateError(Exception):
    """Raised by gate items; reported as its message without a traceback."""


class _GateItem(pytest.Item):
    def repr_failure(
        self,
        excinfo: pytest.ExceptionInfo[BaseException],
        style: str | None = None,
    ) -> str:
        if isinstance(excinfo.value, GateError):
            return str(excinfo.value)
        return str(super().repr_failure(excinfo))

    def reportinfo(self) -> tuple[Path, int | None, str]:
        return self.path, None, self.name


class RateItem(_GateItem):
    def __init__(self, *, check: RateCheck | None, skipped: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.check = check
        self.skipped = skipped

    def runtest(self) -> None:
        if self.check is None:
            pytest.skip(self.skipped or "rate check not computed")
        c = self.check
        if not c.passed:
            raise GateError(
                f"hallucination rate upper bound {c.upper:.3f} ({c.confidence:.0%} one-sided, "
                f"{c.estimator}, estimate {c.estimate:.3f}) exceeds the budget {c.max_rate:.3f}; "
                f"{c.n_labelled} labelled and {c.n_unlabelled} unlabelled responses"
            )


class ResponseItem(_GateItem):
    def __init__(self, *, check: ResponseCheck, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.check = check

    def runtest(self) -> None:
        c = self.check
        if c.flagged:
            detail = (
                f"; least supported unit ({c.unit_score:.2f}): {c.unit_text!r}"
                if c.unit_text
                else ""
            )
            raise GateError(
                f"response {c.id} flagged: calibrated hallucination probability "
                f"{c.probability:.3f} >= threshold {c.threshold:.3f}{detail}"
            )


class GateFile(pytest.File):
    def __init__(self, *, outcome: GateOutcome, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.outcome = outcome

    def collect(self) -> Iterator[pytest.Item]:
        yield RateItem.from_parent(
            self, name="rag_gate_rate", check=self.outcome.rate, skipped=self.outcome.rate_skipped
        )
        for check in self.outcome.responses:
            yield ResponseItem.from_parent(self, name=f"rag_gate_response[{check.id}]", check=check)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    option = config.getoption("rag_gate")
    if not option:
        return
    path = Path(option).resolve()
    try:
        outcome = evaluate(load_config(path))
    except (OSError, ValueError) as exc:
        raise pytest.UsageError(f"--rag-gate {option}: {exc}") from exc
    config.stash[_OUTCOME] = outcome
    gate_file = GateFile.from_parent(session, path=path, outcome=outcome)
    items.extend(gate_file.collect())


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    outcome = config.stash.get(_OUTCOME, None)
    if outcome is None:
        return
    terminalreporter.section("rag-gate")
    if outcome.rate is not None:
        r = outcome.rate
        lam = f", lambda {r.lam:.2f}" if r.lam is not None else ""
        terminalreporter.write_line(
            f"rate ({r.estimator}{lam}): estimate {r.estimate:.3f}, {r.confidence:.0%} upper bound "
            f"{r.upper:.3f}, budget {r.max_rate:.3f} -> {'pass' if r.passed else 'FAIL'} "
            f"[{r.n_labelled} labelled, {r.n_unlabelled} unlabelled]"
        )
    else:
        terminalreporter.write_line(f"rate check skipped: {outcome.rate_skipped}")
    if outcome.responses:
        flagged = sum(c.flagged for c in outcome.responses)
        terminalreporter.write_line(
            f"{flagged} of {len(outcome.responses)} responses at or above the flag threshold "
            f"{outcome.responses[0].threshold:.3f}"
        )
