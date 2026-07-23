from pathlib import Path

import pytest
import yaml

from agenteval.cli import _cmd_import, build_parser
from agenteval.core.dataset_import import (
    DatasetImportError,
    ImportMapping,
    emit_mapping_template,
    import_csv,
    load_mapping,
    write_golden_yaml,
)
from agenteval.core.schema import load_test_cases

BASIC_CSV = """id,question,answer,tools
q1,"What is 2+2?",4,calc
q2,Capital of France?,Paris,
"""

UNICODE_CSV = """id,question,answer
q1,Café ou thé?,Café
q2,日本の首都は?,東京
"""


def write_csv(tmp_path: Path, content: str, name: str = "data.csv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def basic_mapping(**overrides) -> ImportMapping:
    return ImportMapping.from_dict(
        {
            "prompt_column": "question",
            "ground_truth_column": "answer",
            "id_column": "id",
            "correctness_type": "exact",
            "tags": ["imported"],
            "must_call_tools_column": "tools",
            **overrides,
        }
    )


# --- ImportMapping -------------------------------------------------------------


def test_mapping_from_dict_parses_full_config():
    mapping = basic_mapping()
    assert mapping.prompt_column == "question"
    assert mapping.ground_truth_column == "answer"
    assert mapping.id_column == "id"
    assert mapping.tags == ("imported",)
    assert mapping.must_call_tools_column == "tools"


def test_mapping_requires_prompt_and_ground_truth_columns():
    with pytest.raises(DatasetImportError, match="prompt_column is required"):
        ImportMapping.from_dict({"ground_truth_column": "answer"})
    with pytest.raises(DatasetImportError, match="ground_truth_column is required"):
        ImportMapping.from_dict({"prompt_column": "question"})


def test_mapping_rejects_invalid_correctness_type():
    with pytest.raises(DatasetImportError, match="correctness_type must be one of"):
        basic_mapping(correctness_type="not_a_type")


def test_mapping_rejects_numeric_table_correctness_type():
    with pytest.raises(DatasetImportError, match="not supported by CSV import"):
        basic_mapping(correctness_type="numeric_table")


def test_mapping_rejects_non_string_tags():
    with pytest.raises(DatasetImportError, match="tags must be a list of strings"):
        basic_mapping(tags=[1, 2])


def test_mapping_defaults_are_sensible():
    mapping = ImportMapping.from_dict({"prompt_column": "q", "ground_truth_column": "a"})
    assert mapping.id_column is None
    assert mapping.correctness_type == "exact"
    assert mapping.numeric_tolerance == 0.01
    assert mapping.tags == ()
    assert mapping.must_call_tools_column is None


def test_mapping_from_dict_rejects_non_mapping():
    with pytest.raises(DatasetImportError, match="must be a YAML mapping"):
        ImportMapping.from_dict(["not", "a", "dict"])


# --- load_mapping / emit_mapping_template ---------------------------------------


def test_load_mapping_round_trips_through_emit_template(tmp_path):
    path = tmp_path / "mapping.yaml"
    emit_mapping_template(path)
    mapping = load_mapping(path)
    assert mapping.prompt_column == "question"
    assert mapping.ground_truth_column == "answer"
    assert mapping.correctness_type == "exact"


def test_load_mapping_missing_file_raises(tmp_path):
    with pytest.raises(DatasetImportError, match="mapping config not found"):
        load_mapping(tmp_path / "does-not-exist.yaml")


def test_load_mapping_invalid_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("prompt_column: [unterminated\n", encoding="utf-8")
    with pytest.raises(DatasetImportError, match="invalid mapping config YAML"):
        load_mapping(path)


def test_emit_mapping_template_refuses_to_overwrite_without_force(tmp_path):
    path = tmp_path / "mapping.yaml"
    emit_mapping_template(path)
    with pytest.raises(DatasetImportError, match="already exists"):
        emit_mapping_template(path)


def test_emit_mapping_template_force_overwrites(tmp_path):
    path = tmp_path / "mapping.yaml"
    emit_mapping_template(path)
    path_again = emit_mapping_template(path, force=True)
    assert path_again.is_file()


# --- import_csv ------------------------------------------------------------------


def test_import_csv_produces_expected_cases(tmp_path):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    cases = import_csv(csv_path, basic_mapping())
    assert [case.id for case in cases] == ["q1", "q2"]
    assert cases[0].prompt == "What is 2+2?"
    assert cases[0].expects.ground_truth == "4"
    assert cases[0].expects.must_call_tools == ["calc"]
    assert cases[0].tags == ["imported"]
    assert cases[1].expects.must_call_tools == []


def test_import_csv_auto_generates_ids_when_no_id_column(tmp_path):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    mapping = basic_mapping(id_column=None)
    cases = import_csv(csv_path, mapping)
    assert [case.id for case in cases] == ["row_1", "row_2"]


def test_import_csv_missing_column_lists_available_columns(tmp_path):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    mapping = basic_mapping()
    bad_mapping = ImportMapping.from_dict(
        {"prompt_column": "does_not_exist", "ground_truth_column": "answer"}
    )
    with pytest.raises(DatasetImportError, match="missing configured column"):
        import_csv(csv_path, bad_mapping)
    # sanity: the good mapping still works on the same file
    import_csv(csv_path, mapping)


def no_tools_mapping(**overrides) -> ImportMapping:
    return ImportMapping.from_dict(
        {"prompt_column": "question", "ground_truth_column": "answer", "id_column": "id", **overrides}
    )


def test_import_csv_rejects_empty_csv(tmp_path):
    csv_path = write_csv(tmp_path, "id,question,answer\n")
    with pytest.raises(DatasetImportError, match="no data rows"):
        import_csv(csv_path, no_tools_mapping())


def test_import_csv_rejects_blank_prompt(tmp_path):
    csv_path = write_csv(tmp_path, "id,question,answer\nq1,,4\n")
    with pytest.raises(DatasetImportError, match="row 1.*question.*empty"):
        import_csv(csv_path, no_tools_mapping())


def test_import_csv_rejects_blank_ground_truth(tmp_path):
    csv_path = write_csv(tmp_path, "id,question,answer\nq1,hi,\n")
    with pytest.raises(DatasetImportError, match="row 1.*answer.*empty"):
        import_csv(csv_path, no_tools_mapping())


def test_import_csv_rejects_duplicate_ids(tmp_path):
    csv_path = write_csv(tmp_path, "id,question,answer\nq1,a,1\nq1,b,2\n")
    with pytest.raises(DatasetImportError, match="duplicate case id 'q1'"):
        import_csv(csv_path, no_tools_mapping())


def test_import_csv_rejects_non_numeric_ground_truth_for_numeric_type(tmp_path):
    csv_path = write_csv(tmp_path, "id,question,answer\nq1,how many,not-a-number\n")
    mapping = no_tools_mapping(correctness_type="numeric")
    with pytest.raises(DatasetImportError, match="is not numeric"):
        import_csv(csv_path, mapping)


def test_import_csv_accepts_numeric_ground_truth_for_numeric_type(tmp_path):
    csv_path = write_csv(tmp_path, "id,question,answer\nq1,how many,42\n")
    mapping = no_tools_mapping(correctness_type="numeric")
    cases = import_csv(csv_path, mapping)
    assert cases[0].expects.correctness_type.value == "numeric"


def test_import_csv_handles_unicode_content(tmp_path):
    csv_path = write_csv(tmp_path, UNICODE_CSV)
    mapping = ImportMapping.from_dict(
        {"prompt_column": "question", "ground_truth_column": "answer", "id_column": "id"}
    )
    cases = import_csv(csv_path, mapping)
    assert cases[0].prompt == "Café ou thé?"
    assert cases[1].expects.ground_truth == "東京"


def test_import_csv_missing_file_raises(tmp_path):
    with pytest.raises(DatasetImportError, match="CSV file not found"):
        import_csv(tmp_path / "nope.csv", basic_mapping())


# --- write_golden_yaml -------------------------------------------------------------


def test_write_golden_yaml_round_trips_through_load_test_cases(tmp_path):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    cases = import_csv(csv_path, basic_mapping())
    out_path = write_golden_yaml(cases, tmp_path / "imported.yaml")
    loaded = load_test_cases(out_path)
    assert [case.id for case in loaded] == ["q1", "q2"]
    assert loaded[0].expects.ground_truth == "4"
    assert loaded[0].tags == ["imported"]


def test_write_golden_yaml_rejects_empty_case_list(tmp_path):
    with pytest.raises(DatasetImportError, match="no cases to write"):
        write_golden_yaml([], tmp_path / "imported.yaml")


def test_write_golden_yaml_refuses_to_overwrite_without_force(tmp_path):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    cases = import_csv(csv_path, basic_mapping())
    out_path = tmp_path / "imported.yaml"
    write_golden_yaml(cases, out_path)
    with pytest.raises(DatasetImportError, match="already exists"):
        write_golden_yaml(cases, out_path)


def test_write_golden_yaml_force_overwrites(tmp_path):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    cases = import_csv(csv_path, basic_mapping())
    out_path = tmp_path / "imported.yaml"
    write_golden_yaml(cases, out_path)
    written_again = write_golden_yaml(cases, out_path, force=True)
    assert written_again.is_file()


def test_write_golden_yaml_is_valid_yaml_with_header_comment(tmp_path):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    cases = import_csv(csv_path, basic_mapping())
    out_path = write_golden_yaml(cases, tmp_path / "imported.yaml")
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("# Imported from an external dataset")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, list) and len(parsed) == 2


# --- CLI --------------------------------------------------------------------------


def test_cli_import_end_to_end(tmp_path, capsys):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    mapping_path = tmp_path / "mapping.yaml"
    # The emitted template's default column names (question/answer/id) already
    # match BASIC_CSV's headers, so it works unmodified for this end-to-end check.
    emit_mapping_template(mapping_path)
    output_path = tmp_path / "imported.yaml"

    args = build_parser().parse_args(
        [
            "import",
            str(csv_path),
            "--mapping",
            str(mapping_path),
            "--output",
            str(output_path),
        ]
    )
    assert _cmd_import(args) == 0
    out = capsys.readouterr().out
    assert "imported=2" in out
    assert str(output_path) in out
    assert output_path.is_file()


def test_cli_import_emit_mapping_template_alone(tmp_path, capsys):
    template_path = tmp_path / "mapping.yaml"
    args = build_parser().parse_args(
        ["import", "--emit-mapping-template", str(template_path)]
    )
    assert _cmd_import(args) == 0
    assert template_path.is_file()
    assert "mapping_template=" in capsys.readouterr().out


def test_cli_import_requires_csv_path_without_template_flag(capsys):
    args = build_parser().parse_args(["import"])
    assert _cmd_import(args) == 2
    assert "csv_path is required" in capsys.readouterr().err


def test_cli_import_requires_mapping_flag(tmp_path, capsys):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    args = build_parser().parse_args(["import", str(csv_path), "--output", "out.yaml"])
    assert _cmd_import(args) == 2
    assert "--mapping is required" in capsys.readouterr().err


def test_cli_import_requires_output_flag(tmp_path, capsys):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    mapping_path = tmp_path / "mapping.yaml"
    emit_mapping_template(mapping_path)
    args = build_parser().parse_args(
        ["import", str(csv_path), "--mapping", str(mapping_path)]
    )
    assert _cmd_import(args) == 2
    assert "--output is required" in capsys.readouterr().err


def test_cli_import_output_collision_requires_overwrite(tmp_path, capsys):
    csv_path = write_csv(tmp_path, BASIC_CSV)
    mapping_path = tmp_path / "mapping.yaml"
    emit_mapping_template(mapping_path)
    output_path = tmp_path / "imported.yaml"
    output_path.write_text("existing content", encoding="utf-8")

    args = build_parser().parse_args(
        ["import", str(csv_path), "--mapping", str(mapping_path), "--output", str(output_path)]
    )
    assert _cmd_import(args) == 2
    assert "already exists" in capsys.readouterr().err

    args_force = build_parser().parse_args(
        [
            "import",
            str(csv_path),
            "--mapping",
            str(mapping_path),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    )
    assert _cmd_import(args_force) == 0
