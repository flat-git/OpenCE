"""Generator, Reflector, and Curator components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .delta import DeltaBatch
from .llm import LLMClient
from .playbook import Playbook
from .prompts import CURATOR_PROMPT, GENERATOR_PROMPT, REFLECTOR_PROMPT
from .curation_rules import CurationRules


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        debug_path = Path("logs/json_failures.log")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as fh:
            fh.write("----\n")
            fh.write(repr(text))
            fh.write("\n")
        raise ValueError(f"LLM response is not valid JSON: {exc}\n{text}") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return data


def _format_optional(value: Optional[str]) -> str:
    return value or "(none)"


@dataclass
class GeneratorOutput:
    reasoning: str
    final_answer: str
    bullet_ids: List[str]
    raw: Dict[str, Any]


class Generator:
    """Produces trajectories using the current playbook."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = GENERATOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: Playbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorOutput:
        base_prompt = self.prompt_template.format(
            playbook=playbook.as_prompt() or "(empty playbook)",
            reflection=_format_optional(reflection),
            question=question,
            context=_format_optional(context),
        )
        prompt = base_prompt
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                reasoning = str(data.get("reasoning", ""))
                final_answer = str(data.get("final_answer", ""))
                bullet_ids = [
                    str(item)
                    for item in data.get("bullet_ids", [])
                    if isinstance(item, (str, int))
                ]
                return GeneratorOutput(
                    reasoning=reasoning,
                    final_answer=final_answer,
                    bullet_ids=bullet_ids,
                    raw=data,
                )
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                prompt = (
                    base_prompt
                    + "\n\n务必仅输出单个有效 JSON 对象，"
                    "请转义所有引号或改用单引号，避免输出额外文本。"
                )
        raise RuntimeError("Generator failed to produce valid JSON.") from last_error


@dataclass
class BulletTag:
    id: str
    tag: str


@dataclass
class ReflectorOutput:
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag]
    error_categories: List[str]  # New field
    culprit_bullets: List[str]  # New field
    raw: Dict[str, Any]


class Reflector:
    """Extracts lessons and bullet feedback from trajectories."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str],
        feedback: Optional[str],
        diagnostics: Optional[object] = None,  # Diagnostics from environment
        max_refinement_rounds: int = 1,
        **kwargs: Any,
    ) -> ReflectorOutput:
        playbook_excerpt = _make_playbook_excerpt(playbook, generator_output.bullet_ids)
        
        # Add diagnostics to prompt if available
        diagnostics_str = "(no diagnostics)"
        if diagnostics:
            try:
                # Assuming diagnostics has a to_dict method
                diagnostics_dict = diagnostics.to_dict() if hasattr(diagnostics, 'to_dict') else {}
                diagnostics_str = json.dumps(diagnostics_dict, ensure_ascii=False, indent=2)
            except Exception:
                pass
        
        base_prompt = self.prompt_template.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            playbook_excerpt=playbook_excerpt or "(no bullets referenced)",
            diagnostics=diagnostics_str,
        )
        result: Optional[ReflectorOutput] = None
        prompt = base_prompt
        last_error: Optional[Exception] = None
        for round_idx in range(max_refinement_rounds):
            prompt = base_prompt
            for attempt in range(self.max_retries):
                response = self.llm.complete(
                    prompt, refinement_round=round_idx, **kwargs
                )
                try:
                    data = _safe_json_loads(response.text)
                    bullet_tags: List[BulletTag] = []
                    tags_payload = data.get("bullet_tags", [])
                    if isinstance(tags_payload, Sequence):
                        for item in tags_payload:
                            if isinstance(item, dict) and "id" in item and "tag" in item:
                                bullet_tags.append(
                                    BulletTag(
                                        id=str(item["id"]), tag=str(item["tag"]).lower()
                                    )
                                )
                    candidate = ReflectorOutput(
                        reasoning=str(data.get("reasoning", "")),
                        error_identification=str(data.get("error_identification", "")),
                        root_cause_analysis=str(data.get("root_cause_analysis", "")),
                        correct_approach=str(data.get("correct_approach", "")),
                        key_insight=str(data.get("key_insight", "")),
                        bullet_tags=bullet_tags,
                        error_categories=[
                            str(cat) for cat in data.get("error_categories", [])
                            if isinstance(cat, str)
                        ],
                        culprit_bullets=[
                            str(bid) for bid in data.get("culprit_bullets", [])
                            if isinstance(bid, (str, int))
                        ],
                        raw=data,
                    )
                    result = candidate
                    # Early exit if we already have actionable output
                    if bullet_tags or candidate.key_insight:
                        return candidate
                    break
                except ValueError as err:
                    last_error = err
                    if attempt + 1 >= self.max_retries:
                        break
                    prompt = (
                        base_prompt
                        + "\n\n请严格输出有效 JSON，对双引号进行转义，"
                        "不要输出额外解释性文本。"
                    )
        if result is None:
            raise RuntimeError("Reflector failed to produce a result.") from last_error
        return result


@dataclass
class CuratorOutput:
    delta: DeltaBatch
    raw: Dict[str, Any]


class Curator:
    """Transforms reflections into delta updates."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = CURATOR_PROMPT,
        *,
        max_retries: int = 3,
        enable_validation: bool = False,  # Default to False for backward compatibility
        curation_rules: Optional[CurationRules] = None,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.enable_validation = enable_validation
        self.curation_rules = curation_rules or CurationRules() if enable_validation else None

    def curate(
        self,
        *,
        reflection: ReflectorOutput,
        playbook: Playbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> CuratorOutput:
        base_prompt = self.prompt_template.format(
            progress=progress,
            stats=json.dumps(playbook.stats()),
            reflection=json.dumps(reflection.raw, ensure_ascii=False, indent=2),
            playbook=playbook.as_prompt() or "(empty playbook)",
            question_context=question_context,
        )
        prompt = base_prompt
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                delta = DeltaBatch.from_json(data)
                
                # Validate operations if enabled
                if self.enable_validation and self.curation_rules:
                    delta.operations = self.curation_rules.validate_operations(
                        delta.operations, playbook
                    )
                
                return CuratorOutput(delta=delta, raw=data)
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                prompt = (
                    base_prompt
                    + "\n\n提醒：仅输出有效 JSON，所有字符串请转义双引号或改用单引号，"
                    "不要添加额外文本。"
                )
        raise RuntimeError("Curator failed to produce valid JSON.") from last_error


def _make_playbook_excerpt(playbook: Playbook, bullet_ids: Sequence[str]) -> str:
    lines: List[str] = []
    seen = set()
    for bullet_id in bullet_ids:
        if bullet_id in seen:
            continue
        bullet = playbook.get_bullet(bullet_id)
        if bullet:
            seen.add(bullet_id)
            lines.append(f"[{bullet.id}] {bullet.content}")
    return "\n".join(lines)
