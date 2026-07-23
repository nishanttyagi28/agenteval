from agenteval.core.pricing import MODEL_PRICING, get_pricing


def test_get_pricing_falls_back_to_default_when_model_is_none():
    assert get_pricing(None, default=(1.0, 2.0)) == (1.0, 2.0)


def test_get_pricing_falls_back_to_default_for_unknown_model():
    assert get_pricing("some-future-model-nobody-added-yet", default=(1.0, 2.0)) == (1.0, 2.0)


def test_get_pricing_returns_table_entry_when_known():
    assert get_pricing("gpt-4o-mini", default=(1.0, 2.0)) == MODEL_PRICING["gpt-4o-mini"]


def test_model_pricing_table_entries_are_positive_tuples():
    for model, (input_rate, output_rate) in MODEL_PRICING.items():
        assert input_rate > 0, model
        assert output_rate > 0, model
