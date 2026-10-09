from __future__ import annotations

import json

import pytest

from faithgate.judges import (
    AlwaysHallucinatedJudge,
    LLMJudge,
    NLIJudge,
    OverlapJudge,
    build_prompts,
    consistency_rate,
    parse_verdict,
)
from faithgate.llm import Message, ScriptedProvider
from faithgate.records import Record
from faithgate.summac import ChunkConfig, ConvAggregator
from tests.fakes import LexicalEntailment

SOURCE = "The museum opens at ten on weekdays. Adult tickets cost fifteen dollars."
FAITHFUL = Record(id="a", source=SOURCE, response="The museum opens at ten on weekdays.", label=0)
INVENTED = Record(
    id="b",
    source=SOURCE,
    response="The museum opens at ten. A rooftop aquarium offers dolphin shows.",
    label=1,
    spans=[(25, 64)],
)


def test_overlap_judge_scores_and_spans() -> None:
    good, bad = OverlapJudge().score([FAITHFUL, INVENTED])
    assert good.score == 0.0
    assert good.spans == []
    assert bad.score > 0.4
    flagged = [INVENTED.response[a:b] for a, b in bad.spans]
    assert flagged == ["rooftop aquarium offers dolphin shows"]
    assert bad.units[1].score == 1.0


def test_always_hallucinated_flags_everything() -> None:
    [row] = AlwaysHallucinatedJudge().score([INVENTED])
    assert row.score == 1.0
    assert row.spans == [(0, len(INVENTED.response))]


def test_nli_judge_zero_shot_and_conv(lexical_nli: LexicalEntailment) -> None:
    config = ChunkConfig(chunk_words=8)
    judge = NLIJudge(lexical_nli, config, "zs-min", keep_matrix=True)
    good, bad = judge.score([FAITHFUL, INVENTED])
    assert judge.name == "nli-zs-min"
    assert good.score < 0.05
    assert bad.score > 0.9
    assert [INVENTED.response[a:b] for a, b in bad.spans] == [
        "rooftop aquarium offers dolphin shows"
    ]
    assert bad.units[0].evidence is not None
    assert bad.unit_probs is not None
    assert len(bad.unit_probs) == len(bad.units)

    matrices = judge.matrices([FAITHFUL, INVENTED] * 10)
    conv = ConvAggregator(n_bins=5, mode="max").fit(matrices, [0, 1] * 10)
    conv_judge = NLIJudge(lexical_nli, config, conv)
    good_c, bad_c = conv_judge.from_matrices([FAITHFUL, INVENTED], matrices[:2])
    assert conv_judge.name == "nli-conv-max"
    assert bad_c.score > good_c.score


def test_nli_judge_rejects_unknown_aggregation(lexical_nli: LexicalEntailment) -> None:
    with pytest.raises(ValueError, match="aggregation"):
        NLIJudge(lexical_nli, ChunkConfig(), "median")


def _reply(verdict: str, quotes: list[str] | None = None) -> str:
    return "Sure.\n" + json.dumps({"verdict": verdict, "unsupported_quotes": quotes or []})


def test_prompts_swap_block_order() -> None:
    forward, swapped = build_prompts(INVENTED)
    fwd, swp = forward[1]["content"], swapped[1]["content"]
    assert fwd.index("SOURCE:") < fwd.index("RESPONSE:")
    assert swp.index("RESPONSE:") < swp.index("SOURCE:")
    assert forward[0] == swapped[0]


def test_llm_judge_consistent_verdicts_and_quote_spans() -> None:
    provider = ScriptedProvider(lambda m: _reply("unsupported", ["rooftop aquarium"]))
    [row] = LLMJudge(provider).score([INVENTED])
    assert row.score == 1.0
    assert row.consistent is True
    assert [INVENTED.response[a:b] for a, b in row.spans] == ["rooftop aquarium"]
    assert len(provider.calls) == 2


def test_llm_judge_position_bias_becomes_uncertainty() -> None:
    def biased(messages: list[Message]) -> str:
        user = messages[1]["content"]
        source_first = user.index("SOURCE:") < user.index("RESPONSE:")
        return _reply("supported" if source_first else "unsupported")

    rows = LLMJudge(ScriptedProvider(biased)).score([FAITHFUL, INVENTED])
    assert [r.score for r in rows] == [0.5, 0.5]
    assert consistency_rate(rows) == 0.0


def test_llm_judge_unreadable_reply() -> None:
    [row] = LLMJudge(ScriptedProvider(lambda m: "I think it is fine")).score([FAITHFUL])
    assert row.score == 0.5
    assert row.consistent is False


def test_parse_verdict() -> None:
    assert parse_verdict('{"verdict": "Supported"}') == (0.0, [])
    assert parse_verdict('x {"verdict": "unsupported", "unsupported_quotes": ["a", " "]} y') == (
        1.0,
        ["a"],
    )
    assert parse_verdict("{not json}") == (None, [])
    assert parse_verdict('{"verdict": "maybe"}') == (None, [])
