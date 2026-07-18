"""Generate reviewable adversarial variants from hand-written golden cases."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from agenteval.core.config import resolve_agent_repo
from agenteval.core.schema import CorrectnessType, Expects, TestCase

MUTATION_TYPES = {
    "reworded",
    "ambiguous_context",
    "distractor",
    "numeric_precision",
    "mild_injection",
}

ChatFunction = Callable[..., tuple[str | None, str | None]]


def _expects_dict(expects: Expects) -> dict[str, Any]:
    data = asdict(expects)
    correctness_type = data.get("correctness_type")
    if isinstance(correctness_type, CorrectnessType):
        data["correctness_type"] = correctness_type.value
    return data


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        raise ValueError("generator returned an empty response")
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        raise ValueError("generator response does not contain a JSON array")
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"generator returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("generator JSON must be an array of objects")
    return parsed


def _default_chat() -> ChatFunction:
    agent_repo = resolve_agent_repo()
    if str(agent_repo) not in sys.path:
        sys.path.insert(0, str(agent_repo))
    from utils.env import load_project_env

    load_project_env()
    from agents.llm_client import chat_completion

    return chat_completion


def generator_messages(case: TestCase, variants: int) -> list[dict[str, str]]:
    allowed = ", ".join(sorted(MUTATION_TYPES))
    expectations = json.dumps(_expects_dict(case.expects), ensure_ascii=False)
    system = (
        "You generate adversarial test prompts for an AI data-analysis agent. "
        "Every variant MUST preserve the exact question meaning, dataset scope, ground truth, "
        "and expected tool route. Do not introduce facts that change the correct answer. "
        "Ambiguity may add distracting wording but must remain answerable. Mild injection may "
        "ask the agent to guess or skip tools, while the original correct answer remains valid. "
        "Return JSON only: an array of objects with mutation_type and prompt."
    )
    user = (
        f"Create exactly {variants} distinct variants.\n"
        f"Allowed mutation types: {allowed}\n"
        f"Original id: {case.id}\nOriginal prompt: {case.prompt}\n"
        f"Immutable expectations: {expectations}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_variants(
    case: TestCase,
    *,
    variants: int = 3,
    chat: ChatFunction | None = None,
) -> list[TestCase]:
    if variants < 1:
        raise ValueError("variants must be at least 1")
    chat_fn = chat or _default_chat()
    response, error = chat_fn(
        generator_messages(case, variants),
        temperature=0.4,
        max_tokens=max(700, variants * 220),
    )
    if error:
        raise RuntimeError(f"adversarial generator error for {case.id}: {error}")
    items = _extract_json_array(response or "")
    if len(items) != variants:
        raise ValueError(
            f"generator returned {len(items)} variants for {case.id}; expected {variants}"
        )

    generated: list[TestCase] = []
    seen_prompts = {case.prompt.strip().lower()}
    for index, item in enumerate(items, start=1):
        mutation = str(item.get("mutation_type") or "").strip().lower()
        prompt = str(item.get("prompt") or "").strip()
        if mutation not in MUTATION_TYPES:
            raise ValueError(f"unsupported mutation_type {mutation!r} for {case.id}")
        if not prompt:
            raise ValueError(f"empty adversarial prompt for {case.id}")
        normalized = re.sub(r"\s+", " ", prompt.lower())
        if normalized in seen_prompts:
            raise ValueError(f"duplicate adversarial prompt for {case.id}")
        seen_prompts.add(normalized)
        generated.append(
            TestCase(
                id=f"{case.id}__{mutation}__{index:02d}",
                prompt=prompt,
                expects=Expects.from_dict(_expects_dict(case.expects)),
                tags=list(dict.fromkeys([*case.tags, "adversarial", mutation])),
                source="adversarial",
                parent_id=case.id,
                mutation_type=mutation,
                review_status="candidate",
            )
        )
    return generated


def generate_suite(
    cases: Sequence[TestCase],
    *,
    variants_per_case: int = 3,
    chat: ChatFunction | None = None,
) -> list[TestCase]:
    generated: list[TestCase] = []
    for case in cases:
        generated.extend(generate_variants(case, variants=variants_per_case, chat=chat))
    return generated


def test_case_to_dict(case: TestCase) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": case.id,
        "prompt": case.prompt,
        "expects": _expects_dict(case.expects),
        "tags": case.tags,
    }
    for key in ("source", "parent_id", "mutation_type", "review_status"):
        value = getattr(case, key)
        if value is not None:
            data[key] = value
    return data


def write_candidate_yaml(cases: Sequence[TestCase], path: str | Path) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write adversarial cases") from exc
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [test_case_to_dict(case) for case in cases]
    output.write_text(
        "# Generated candidates: review before moving into the active adversarial suite.\n"
        + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output
