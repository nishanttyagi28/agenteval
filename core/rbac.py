"""Role-based access control -- schema and pure permission logic (Tier 7, Phase 2).

Deliberately decoupled from any real authentication or hosting: this
module answers "can this role do X", never "who is this request actually
from" -- that identity-verification piece gets wired to a real auth
provider once real hosting exists (see the README's Deployment section).
Loading an ``rbac.yaml`` is entirely optional; nothing else in AgentEval
requires it to exist, and no command currently checks permissions
automatically -- this is the groundwork a future hosted deployment wires
itself around, not an enforcement layer active today.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Role(str, Enum):
    admin = "admin"
    contributor = "contributor"
    viewer = "viewer"


class Permission(str, Enum):
    view_runs = "view_runs"
    trigger_run = "trigger_run"
    modify_config = "modify_config"
    view_audit_log = "view_audit_log"


# admin: everything. contributor: can view and trigger runs, and read the
# audit trail, but not change agent/gate/alerting configuration. viewer:
# read-only.
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.admin: frozenset(Permission),
    Role.contributor: frozenset(
        {Permission.view_runs, Permission.trigger_run, Permission.view_audit_log}
    ),
    Role.viewer: frozenset({Permission.view_runs}),
}


def has_permission(role: Role, permission: Permission) -> bool:
    """Pure role -> permission lookup; no notion of "who" is asking."""
    return permission in ROLE_PERMISSIONS.get(role, frozenset())


@dataclass(frozen=True)
class RbacConfig:
    """A loaded username -> role mapping."""

    users: dict[str, Role]

    def role_for(self, username: str) -> Role | None:
        return self.users.get(username)

    def can(self, username: str, permission: Permission) -> bool:
        """False for an unrecognized user -- unknown means no access, not admin."""
        role = self.role_for(username)
        if role is None:
            return False
        return has_permission(role, permission)


def load_rbac_config(path: str | Path) -> RbacConfig:
    """Load an optional ``rbac.yaml``: ``{version: 1, users: {name: role}}``."""
    import yaml

    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid RBAC YAML in {p}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"RBAC config must be a mapping: {p}")
    if raw.get("version") != 1:
        raise ValueError(f"Unsupported RBAC config version {raw.get('version')!r}; expected 1")

    users_raw: Any = raw.get("users") or {}
    if not isinstance(users_raw, dict):
        raise ValueError("rbac.yaml 'users' must be a mapping of username -> role")

    users: dict[str, Role] = {}
    for username, role_name in users_raw.items():
        if not isinstance(username, str) or not username.strip():
            raise ValueError("rbac.yaml usernames must be non-empty strings")
        try:
            users[username] = Role(role_name)
        except ValueError as exc:
            valid = ", ".join(role.value for role in Role)
            raise ValueError(
                f"rbac.yaml: invalid role {role_name!r} for user {username!r}; expected one of {valid}"
            ) from exc
    return RbacConfig(users=users)
