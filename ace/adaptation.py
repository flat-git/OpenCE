"""Adaptation loops for offline and online ACE training."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .deduplication import Deduplicator
from .diagnostics import Diagnostics
from .gating import BulletGate, GatingConfig
from .playbook import Playbook
from .roles import Curator, CuratorOutput, Generator, GeneratorOutput, Reflector, ReflectorOutput


@dataclass
class Sample:
    """Single task instance presented to ACE."""

    question: str
    context: str = ""
    ground_truth: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EnvironmentResult:
    """Feedback returned by the task environment after executing the generator output."""

    feedback: str
    ground_truth: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    diagnostics: Optional[Diagnostics] = None


class TaskEnvironment(ABC):
    """Defines how to evaluate generator outputs for a sample."""

    @abstractmethod
    def evaluate(
        self, sample: Sample, generator_output: GeneratorOutput
    ) -> EnvironmentResult:
        """Return environment feedback plus optional ground truth information."""


@dataclass
class AdapterStepResult:
    sample: Sample
    generator_output: GeneratorOutput
    environment_result: EnvironmentResult
    reflection: ReflectorOutput
    curator_output: CuratorOutput
    playbook_snapshot: str


class AdapterBase:
    """Shared orchestration logic for offline and online ACE adaptation."""

    def __init__(
        self,
        *,
        playbook: Optional[Playbook] = None,
        generator: Generator,
        reflector: Reflector,
        curator: Curator,
        max_refinement_rounds: int = 1,
        reflection_window: int = 20,  # Updated from 3 to 20
        enable_gating: bool = True,
        gating_config: Optional[GatingConfig] = None,
    ) -> None:
        self.playbook = playbook or Playbook()
        self.generator = generator
        self.reflector = reflector
        self.curator = curator
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window
        self.enable_gating = enable_gating
        self.bullet_gate = BulletGate(gating_config) if enable_gating else None
        self._recent_reflections: List[str] = []

    # ------------------------------------------------------------------ #
    def _reflection_context(self) -> str:
        return "\n---\n".join(self._recent_reflections)

    def _update_recent_reflections(self, reflection: ReflectorOutput) -> None:
        serialized = json.dumps(reflection.raw, ensure_ascii=False)
        self._recent_reflections.append(serialized)
        if len(self._recent_reflections) > self.reflection_window:
            self._recent_reflections = self._recent_reflections[-self.reflection_window :]

    def _apply_bullet_tags(self, reflection: ReflectorOutput) -> None:
        for tag in reflection.bullet_tags:
            try:
                self.playbook.tag_bullet(tag.id, tag.tag)
            except ValueError:
                continue
    
    def _update_bullet_scores(
        self,
        env_result: EnvironmentResult,
        generator_output: GeneratorOutput,
        sample: Sample,
    ) -> None:
        """
        Update helpful/harmful counts and relevance scores for bullets.
        
        Uses diagnostics to determine decision class and updates bullets accordingly.
        """
        if not env_result.diagnostics:
            return
        
        diagnostics = env_result.diagnostics
        decision_class = self._classify_decision(diagnostics)
        
        # Get bullets that influenced this decision
        decision_bullets = diagnostics.decision_basis_bullets or generator_output.bullet_ids
        
        # Calculate relevance if gating is enabled
        relevance = 1.0
        if self.bullet_gate and sample.question:
            # Average relevance across used bullets
            relevances = []
            for bullet_id in decision_bullets:
                bullet = self.playbook.get_bullet(bullet_id)
                if bullet:
                    rel = self.bullet_gate.compute_similarity(sample.question, bullet.content)
                    relevances.append(rel)
            if relevances:
                relevance = sum(relevances) / len(relevances)
        
        # Update helpful/harmful based on decision class
        if decision_class == "POSITIVE":
            # Correct positive: increment helpful for used bullets
            for bullet_id in decision_bullets:
                bullet = self.playbook.tag_bullet(bullet_id, "helpful", increment=1)
                if bullet:
                    bullet.relevance_score = (bullet.helpful - bullet.harmful) * relevance
        
        elif decision_class == "NEGATIVE":
            # Check if this is a false negative (high coverage but marked negative)
            if diagnostics.coverage >= 0.5:
                # Potentially harmful: bullets caused over-conservative rejection
                for bullet_id in decision_bullets:
                    bullet = self.playbook.tag_bullet(bullet_id, "harmful", increment=1)
                    if bullet:
                        bullet.relevance_score = (bullet.helpful - bullet.harmful) * relevance
            # For true negatives with low coverage, don't update (neutral)
        
        elif decision_class == "UNCERTAIN":
            # Mark as neutral
            for bullet_id in decision_bullets:
                bullet = self.playbook.tag_bullet(bullet_id, "neutral", increment=1)
    
    def _classify_decision(self, diagnostics: Diagnostics) -> str:
        """
        Classify decision into POSITIVE/UNCERTAIN/NEGATIVE based on diagnostics.
        
        Rules:
        - POSITIVE: core elements explicitly satisfied AND coverage >= 0.7
        - UNCERTAIN: 0.4 <= coverage < 0.7 OR core elements functional match
        - NEGATIVE: coverage < 0.4 OR core elements missing
        """
        coverage = diagnostics.coverage
        
        # Check core elements
        core_elements = [e for e in diagnostics.elements if e.is_core]
        if not core_elements:
            # No core elements defined, use coverage only
            if coverage >= 0.7:
                return "POSITIVE"
            elif coverage >= 0.4:
                return "UNCERTAIN"
            else:
                return "NEGATIVE"
        
        # Check if all core elements are explicitly satisfied
        core_explicit = all(e.match == "explicit" for e in core_elements)
        core_functional = any(e.match == "functional" for e in core_elements)
        core_missing = any(e.match == "none" for e in core_elements)
        
        if core_explicit and coverage >= 0.7:
            return "POSITIVE"
        elif core_missing or coverage < 0.4:
            return "NEGATIVE"
        else:
            return "UNCERTAIN"

    def _question_context(self, sample: Sample, environment_result: EnvironmentResult) -> str:
        parts = [
            f"question: {sample.question}",
            f"context: {sample.context}",
            f"metadata: {json.dumps(sample.metadata)}",
            f"feedback: {environment_result.feedback}",
            f"ground_truth: {environment_result.ground_truth}",
        ]
        return "\n".join(parts)

    def _progress_string(self, epoch: int, total_epochs: int, step: int, total_steps: int) -> str:
        return f"epoch {epoch}/{total_epochs} · sample {step}/{total_steps}"

    def _process_sample(
        self,
        sample: Sample,
        environment: TaskEnvironment,
        *,
        epoch: int,
        total_epochs: int,
        step_index: int,
        total_steps: int,
    ) -> AdapterStepResult:
        generator_output = self.generator.generate(
            question=sample.question,
            context=sample.context,
            playbook=self.playbook,
            reflection=self._reflection_context(),
        )
        env_result = environment.evaluate(sample, generator_output)
        
        # Update bullet scores based on diagnostics
        self._update_bullet_scores(env_result, generator_output, sample)
        
        reflection = self.reflector.reflect(
            question=sample.question,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=env_result.ground_truth,
            feedback=env_result.feedback,
            diagnostics=env_result.diagnostics,
            max_refinement_rounds=self.max_refinement_rounds,
        )
        self._apply_bullet_tags(reflection)
        self._update_recent_reflections(reflection)
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=self._question_context(sample, env_result),
            progress=self._progress_string(epoch, total_epochs, step_index, total_steps),
        )
        self.playbook.apply_delta(curator_output.delta)
        return AdapterStepResult(
            sample=sample,
            generator_output=generator_output,
            environment_result=env_result,
            reflection=reflection,
            curator_output=curator_output,
            playbook_snapshot=self.playbook.as_prompt(),
        )


class OfflineAdapter(AdapterBase):
    """Runs multi-epoch offline adaptation on a training split."""

    def __init__(
        self,
        *,
        playbook: Optional[Playbook] = None,
        generator: Generator,
        reflector: Reflector,
        curator: Curator,
        deduplicator: Optional[Deduplicator] = None,
        max_refinement_rounds: int = 1,
        reflection_window: int = 20,
        enable_gating: bool = False,  # Default to False for backward compatibility
        gating_config: Optional[GatingConfig] = None,
    ) -> None:
        super().__init__(
            playbook=playbook,
            generator=generator,
            reflector=reflector,
            curator=curator,
            max_refinement_rounds=max_refinement_rounds,
            reflection_window=reflection_window,
            enable_gating=enable_gating,
            gating_config=gating_config,
        )
        self.deduplicator = deduplicator

    def run(
        self,
        samples: Sequence[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
    ) -> List[AdapterStepResult]:
        results: List[AdapterStepResult] = []
        total_steps = len(samples)
        for epoch_idx in range(1, epochs + 1):
            bullet_ids_this_epoch = []
            for step_idx, sample in enumerate(samples, start=1):
                result = self._process_sample(
                    sample,
                    environment,
                    epoch=epoch_idx,
                    total_epochs=epochs,
                    step_index=step_idx,
                    total_steps=total_steps,
                )
                results.append(result)
                bullet_ids_this_epoch.extend(
                    op.bullet_id for op in result.curator_output.delta.operations if op.bullet_id
                )

            if self.deduplicator:
                self.playbook.deduplicate(self.deduplicator, bullet_ids_this_epoch)

        return results


class OnlineAdapter(AdapterBase):
    """Processes a stream of samples sequentially, updating the playbook in-place."""

    def run(
        self,
        samples: Iterable[Sample],
        environment: TaskEnvironment,
    ) -> List[AdapterStepResult]:
        results: List[AdapterStepResult] = []
        step_idx = 0
        for step_idx, sample in enumerate(samples, start=1):
            result = self._process_sample(
                sample,
                environment,
                epoch=1,
                total_epochs=1,
                step_index=step_idx,
                total_steps=step_idx,
            )
            results.append(result)
        return results
