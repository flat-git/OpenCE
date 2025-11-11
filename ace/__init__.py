"""Agentic Context Engineering (ACE) reproduction framework."""

from .playbook import Bullet, Playbook
from .delta import DeltaOperation, DeltaBatch
from .diagnostics import Diagnostics, ElementExtractor, ElementMatch
from .gating import BulletGate, GatingConfig
from .curation_rules import CurationRules
from .reporting import PlaybookReporter, BulletReport
from .llm import LLMClient, DummyLLMClient, TransformersLLMClient
from .llm_deepseek import DeepSeekClient
from .roles import (
    Generator,
    Reflector,
    Curator,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
    BulletTag,
)
from .adaptation import (
    OfflineAdapter,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    AdapterStepResult,
)

__all__ = [
    "Bullet",
    "Playbook",
    "DeltaOperation",
    "DeltaBatch",
    "Diagnostics",
    "ElementExtractor",
    "ElementMatch",
    "BulletGate",
    "GatingConfig",
    "CurationRules",
    "PlaybookReporter",
    "BulletReport",
    "LLMClient",
    "DummyLLMClient",
    "TransformersLLMClient",
    "DeepSeekClient",
    "Generator",
    "Reflector",
    "Curator",
    "GeneratorOutput",
    "ReflectorOutput",
    "CuratorOutput",
    "BulletTag",
    "OfflineAdapter",
    "OnlineAdapter",
    "Sample",
    "TaskEnvironment",
    "EnvironmentResult",
    "AdapterStepResult",
]
