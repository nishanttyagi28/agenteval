from agenteval.core.flakiness import (
    CaseFlakiness,
    FlakinessReport,
    FlakinessSummary,
)
from agenteval.core.store import save_flakiness_report
from agenteval.dashboard.app import (
    dashboard_tab_labels,
    flakiness_table_rows,
    latest_flakiness_report,
    load_flakiness_runs,
)


def report(run_id: str, *, agent: str = "alpha") -> FlakinessReport:
    case = CaseFlakiness(
        case_id="total_customers",
        classification="flaky",
        consistency_score=0.8,
        consistent_observations=4,
        total_observations=5,
        pass_count=4,
        comparison_basis="verdict_and_numeric_majority_cluster",
    )
    return FlakinessReport(
        run_id=run_id,
        agent=agent,
        repeat_count=5,
        summary=FlakinessSummary(
            cases_evaluated=1,
            stable_cases=0,
            flaky_cases=1,
            unstable_cases=0,
            mean_consistency=0.8,
            additional_invocations=4,
            additional_latency_ms=40.0,
            additional_cost_usd=0.04,
        ),
        cases=(case,),
    )


def test_agent_with_flakiness_data_loads_latest_and_assembles_rows(tmp_path):
    runs = tmp_path / "runs"
    older = report("20260721T120000Z_old")
    newer = report("20260721T130000Z_new")
    save_flakiness_report(newer, runs_root=runs)
    save_flakiness_report(older, runs_root=runs)

    loaded = load_flakiness_runs("alpha", runs_root=runs)
    assert [item[1].run_id for item in loaded] == [newer.run_id, older.run_id]
    latest = latest_flakiness_report("alpha", runs_root=runs)
    assert latest == newer
    assert flakiness_table_rows(latest) == [
        {
            "case_id": "total_customers",
            "consistency": "4/5",
            "pass_rate": "4/5",
            "classification": "flaky",
            "comparison_basis": "verdict_and_numeric_majority_cluster",
        }
    ]


def test_agent_without_flakiness_data_has_no_report(tmp_path):
    runs = tmp_path / "runs"
    save_flakiness_report(report("20260721T120000Z_alpha"), runs_root=runs)
    assert latest_flakiness_report("beta", runs_root=runs) is None
    assert load_flakiness_runs("beta", runs_root=runs) == []
    assert dashboard_tab_labels(None) == [
        "1 · Latest summary",
        "2 · Regression",
        "3 · Case drill-down",
        "4 · Adversarial robustness",
    ]


def test_flakiness_report_adds_only_the_fifth_tab():
    labels = dashboard_tab_labels(report("20260721T120000Z_alpha"))
    assert labels[:4] == dashboard_tab_labels(None)
    assert labels[4:] == ["5 · Flakiness"]


def test_corrupt_sidecar_is_silently_ignored(tmp_path):
    directory = tmp_path / "runs" / "alpha" / "flakiness"
    directory.mkdir(parents=True)
    (directory / "broken.json").write_text("not-json", encoding="utf-8")
    assert latest_flakiness_report("alpha", runs_root=tmp_path / "runs") is None
