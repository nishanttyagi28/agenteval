from streamlit.testing.v1 import AppTest

from agenteval.dashboard.app import trajectory_table_rows


def trajectory_payload():
    return {
        "expected": ["route:sql", "agent:sql"],
        "actual": ["route:sql", "planner", "agent:sql"],
        "matched": ["route:sql", "agent:sql"],
        "missing": [],
        "extra": ["planner"],
        "precision": 2 / 3,
        "recall": 1.0,
        "score": 0.8,
        "exact_match": False,
        "order_preserved": True,
    }


def trajectory_app():
    from agenteval.dashboard.app import render_trajectory

    render_trajectory(
        {
            "expected": ["route:sql", "agent:sql"],
            "actual": ["route:sql", "planner", "agent:sql"],
            "matched": ["route:sql", "agent:sql"],
            "missing": [],
            "extra": ["planner"],
            "precision": 2 / 3,
            "recall": 1.0,
            "score": 0.8,
            "exact_match": False,
            "order_preserved": True,
        }
    )


def no_trajectory_app():
    from agenteval.dashboard.app import render_trajectory

    render_trajectory(None)


def test_trajectory_table_preserves_positions_without_inventing_alignment():
    assert trajectory_table_rows(trajectory_payload()) == [
        {"Position": 1, "Expected": "route:sql", "Actual": "route:sql"},
        {"Position": 2, "Expected": "agent:sql", "Actual": "planner"},
        {"Position": 3, "Expected": "—", "Actual": "agent:sql"},
    ]


def test_trajectory_evidence_renders_inside_case_view_components():
    app = AppTest.from_function(trajectory_app, default_timeout=10).run()

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics == {
        "Trajectory score": "80.0%",
        "Trajectory precision": "66.7%",
        "Trajectory recall": "100.0%",
        "Exact match": "No",
    }
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.to_dict(orient="records") == trajectory_table_rows(
        trajectory_payload()
    )
    assert any("#### Trajectory" in item.value for item in app.markdown)
    assert any("Order preserved: yes" in item.value for item in app.caption)


def test_case_without_trajectory_renders_nothing_extra():
    app = AppTest.from_function(no_trajectory_app, default_timeout=10).run()

    assert not app.exception
    assert len(app.metric) == 0
    assert len(app.dataframe) == 0
    assert not any("Trajectory" in item.value for item in app.markdown)
    assert len(app.caption) == 0
