from agenteval.core.metrics import GROQ_INPUT_USD_PER_1M, GROQ_OUTPUT_USD_PER_1M, compute_cost_usd, score_case
from agenteval.core.pricing import MODEL_PRICING
from agenteval.core.schema import CaseResult, CorrectnessType, Expects, TestCase


def numeric_case(case_id="n"):
    return TestCase(
        id=case_id,
        prompt="Return the value",
        expects=Expects(correctness_type=CorrectnessType.numeric, ground_truth=1.0, numeric_tolerance=0.5),
    )


# ── compute_cost_usd ─────────────────────────────────────────────────────────


def test_compute_cost_usd_defaults_to_groq_pricing_when_model_omitted():
    cost, pt, ct, estimated = compute_cost_usd(1_000_000, 1_000_000, "", "")

    assert cost == GROQ_INPUT_USD_PER_1M + GROQ_OUTPUT_USD_PER_1M
    assert (pt, ct, estimated) == (1_000_000, 1_000_000, False)


def test_compute_cost_usd_uses_model_specific_pricing_when_known():
    input_rate, output_rate = MODEL_PRICING["gpt-4o-mini"]

    cost, _, _, _ = compute_cost_usd(1_000_000, 1_000_000, "", "", model="gpt-4o-mini")

    assert cost == input_rate + output_rate
    assert cost != GROQ_INPUT_USD_PER_1M + GROQ_OUTPUT_USD_PER_1M


def test_compute_cost_usd_falls_back_to_groq_pricing_for_unknown_model():
    cost, _, _, _ = compute_cost_usd(1_000_000, 1_000_000, "", "", model="totally-unknown-model")

    assert cost == GROQ_INPUT_USD_PER_1M + GROQ_OUTPUT_USD_PER_1M


# ── score_case reads the model from raw["_llm_usage"]["model"] ─────────────


def test_score_case_uses_model_from_llm_usage_metadata():
    case = numeric_case()
    input_rate, output_rate = MODEL_PRICING["gpt-4o-mini"]
    result = score_case(
        case,
        CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            final_answer="1.0",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            raw={"_llm_usage": {"model": "gpt-4o-mini"}},
        ),
    )

    assert result.cost_usd == input_rate + output_rate


def test_score_case_ignores_non_string_model_field():
    case = numeric_case()
    result = score_case(
        case,
        CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            final_answer="1.0",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            raw={"_llm_usage": {"model": 12345}},
        ),
    )

    assert result.cost_usd == GROQ_INPUT_USD_PER_1M + GROQ_OUTPUT_USD_PER_1M


def test_score_case_without_llm_usage_metadata_uses_default_pricing():
    case = numeric_case()
    result = score_case(
        case,
        CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            final_answer="1.0",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        ),
    )

    assert result.cost_usd == GROQ_INPUT_USD_PER_1M + GROQ_OUTPUT_USD_PER_1M


def test_score_case_uses_model_pricing_on_agent_error_path_too():
    case = numeric_case()
    input_rate, output_rate = MODEL_PRICING["gpt-4o-mini"]
    result = score_case(
        case,
        CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            final_answer="",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            raw={
                "success": False,
                "error": "boom",
                "_llm_usage": {"model": "gpt-4o-mini"},
            },
        ),
    )

    assert result.status == "agent_error"
    assert result.cost_usd == input_rate + output_rate


def test_score_case_uses_model_pricing_on_harness_error_path_too():
    case = numeric_case()
    input_rate, output_rate = MODEL_PRICING["gpt-4o-mini"]
    result = score_case(
        case,
        CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            final_answer="",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            raw={
                "success": False,
                "error": "boom",
                "route": "harness_error",
                "_llm_usage": {"model": "gpt-4o-mini"},
            },
        ),
    )

    assert result.status == "agent_error"
    assert result.cost_usd == input_rate + output_rate
