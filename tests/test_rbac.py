from pathlib import Path

import pytest

from agenteval.core.rbac import (
    Permission,
    RbacConfig,
    Role,
    has_permission,
    load_rbac_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RBAC_EXAMPLE = REPO_ROOT / "rbac.example.yaml"

# ── has_permission / ROLE_PERMISSIONS ───────────────────────────────────────


def test_admin_has_every_permission():
    for permission in Permission:
        assert has_permission(Role.admin, permission)


def test_viewer_can_only_view_runs():
    assert has_permission(Role.viewer, Permission.view_runs)
    assert not has_permission(Role.viewer, Permission.trigger_run)
    assert not has_permission(Role.viewer, Permission.modify_config)
    assert not has_permission(Role.viewer, Permission.view_audit_log)


def test_contributor_can_view_and_trigger_but_not_modify_config():
    assert has_permission(Role.contributor, Permission.view_runs)
    assert has_permission(Role.contributor, Permission.trigger_run)
    assert has_permission(Role.contributor, Permission.view_audit_log)
    assert not has_permission(Role.contributor, Permission.modify_config)


def test_only_admin_can_modify_config():
    assert has_permission(Role.admin, Permission.modify_config)
    assert not has_permission(Role.contributor, Permission.modify_config)
    assert not has_permission(Role.viewer, Permission.modify_config)


# ── RbacConfig ───────────────────────────────────────────────────────────────


def test_role_for_unknown_user_is_none():
    config = RbacConfig(users={"alice": Role.admin})
    assert config.role_for("bob") is None


def test_can_is_false_for_unknown_user_on_every_permission():
    config = RbacConfig(users={"alice": Role.admin})
    for permission in Permission:
        assert config.can("bob", permission) is False


def test_can_delegates_to_has_permission_for_known_user():
    config = RbacConfig(users={"alice": Role.viewer})
    assert config.can("alice", Permission.view_runs) is True
    assert config.can("alice", Permission.trigger_run) is False


# ── load_rbac_config ─────────────────────────────────────────────────────────


def write_rbac_yaml(tmp_path, content):
    path = tmp_path / "rbac.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_rbac_config_parses_valid_file(tmp_path):
    path = write_rbac_yaml(
        tmp_path,
        "version: 1\nusers:\n  alice: admin\n  bob: contributor\n  carol: viewer\n",
    )
    config = load_rbac_config(path)
    assert config.role_for("alice") == Role.admin
    assert config.role_for("bob") == Role.contributor
    assert config.role_for("carol") == Role.viewer


def test_load_rbac_config_with_no_users_key(tmp_path):
    path = write_rbac_yaml(tmp_path, "version: 1\n")
    config = load_rbac_config(path)
    assert config.users == {}


def test_load_rbac_config_rejects_missing_version(tmp_path):
    path = write_rbac_yaml(tmp_path, "users:\n  alice: admin\n")
    with pytest.raises(ValueError, match="Unsupported RBAC config version"):
        load_rbac_config(path)


def test_load_rbac_config_rejects_wrong_version(tmp_path):
    path = write_rbac_yaml(tmp_path, "version: 2\nusers:\n  alice: admin\n")
    with pytest.raises(ValueError, match="Unsupported RBAC config version"):
        load_rbac_config(path)


def test_load_rbac_config_rejects_non_mapping_root(tmp_path):
    path = write_rbac_yaml(tmp_path, "- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_rbac_config(path)


def test_load_rbac_config_rejects_non_mapping_users(tmp_path):
    path = write_rbac_yaml(tmp_path, "version: 1\nusers: not_a_mapping\n")
    with pytest.raises(ValueError, match="'users' must be a mapping"):
        load_rbac_config(path)


def test_load_rbac_config_rejects_invalid_role_name(tmp_path):
    path = write_rbac_yaml(tmp_path, "version: 1\nusers:\n  alice: superuser\n")
    with pytest.raises(ValueError, match="invalid role 'superuser' for user 'alice'"):
        load_rbac_config(path)


def test_load_rbac_config_rejects_blank_username(tmp_path):
    path = write_rbac_yaml(tmp_path, 'version: 1\nusers:\n  "": admin\n')
    with pytest.raises(ValueError, match="usernames must be non-empty strings"):
        load_rbac_config(path)


def test_load_rbac_config_invalid_yaml_raises_value_error(tmp_path):
    path = write_rbac_yaml(tmp_path, "version: 1\nusers: [unterminated\n")
    with pytest.raises(ValueError, match="Invalid RBAC YAML"):
        load_rbac_config(path)


def test_load_rbac_config_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        load_rbac_config(tmp_path / "does_not_exist.yaml")


def test_repo_root_rbac_example_loads_and_covers_every_role():
    config = load_rbac_config(RBAC_EXAMPLE)
    assert set(config.users.values()) == set(Role)
