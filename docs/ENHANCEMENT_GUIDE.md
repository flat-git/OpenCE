# ACE Enhancement Usage Guide

This guide demonstrates how to use the new features added to the ACE framework for improved recall and convergence.

## Overview

The enhanced ACE framework includes:

1. **Bullet Gating**: Similarity-based selection of relevant bullets
2. **Element Diagnostics**: Structured element-level analysis
3. **Three-Class Decision**: POSITIVE/UNCERTAIN/NEGATIVE classification
4. **Helpful/Harmful Tracking**: Bullet performance metrics
5. **Constrained Curation**: Operation validation and limits
6. **Performance Reporting**: Analysis and deprecation recommendations

## Quick Start

### Basic Setup with Enhancements

```python
from ace import (
    OfflineAdapter,
    Generator,
    Reflector,
    Curator,
    Playbook,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Diagnostics,
    ElementExtractor,
    ElementMatch,
    GatingConfig,
    CurationRules,
    PlaybookReporter,
)
from ace import DummyLLMClient  # Replace with your LLM client

# Initialize components with enhancements
playbook = Playbook()
llm_client = DummyLLMClient()

generator = Generator(llm_client)
reflector = Reflector(llm_client)
curator = Curator(llm_client, enable_validation=True)

# Configure gating
gating_config = GatingConfig(
    top_k=25,  # Select top 25 bullets
    guard_strategies=5,  # Plus 5 guard bullets
    similarity_threshold=0.1,
)

# Create adapter with enhancements enabled
adapter = OfflineAdapter(
    playbook=playbook,
    generator=generator,
    reflector=reflector,
    curator=curator,
    enable_gating=True,  # Enable bullet gating
    gating_config=gating_config,
    reflection_window=20,  # Increased from default 3
)
```

## Feature Details

### 1. Bullet Gating

Select only relevant bullets based on similarity to reduce context size:

```python
from ace import BulletGate, GatingConfig

# Configure gating
config = GatingConfig(
    top_k=25,
    guard_strategies=5,
    similarity_threshold=0.1,
)

gate = BulletGate(config, auto_load_encoder=True)

# Get bullets from playbook
bullets = [(b.id, b.content) for b in playbook.bullets()]

# Select relevant bullets for a sample
sample_text = "Question about temperature sensors"
selected_ids = gate.select_bullets(sample_text, bullets)
```

### 2. Element Diagnostics

Track element-level matching for better error analysis:

```python
from ace import ElementExtractor, Diagnostics, ElementMatch

class MyEnvironment(TaskEnvironment):
    def __init__(self):
        self.extractor = ElementExtractor()
    
    def evaluate(self, sample, generator_output):
        # Extract elements from generator output
        elements_list = generator_output.raw.get("elements", [])
        
        # Match each element against evidence
        element_matches = []
        for i, elem in enumerate(elements_list):
            match = self.extractor.match_element(
                elem,
                sample.context,
                is_core=True  # Mark as core element
            )
            match.id = f"E{i+1}"
            element_matches.append(match)
        
        # Compute coverage
        coverage = self.extractor.compute_coverage(element_matches)
        
        # Create diagnostics
        diagnostics = Diagnostics(
            elements=element_matches,
            coverage=coverage,
            decision_basis_bullets=generator_output.bullet_ids,
        )
        
        return EnvironmentResult(
            feedback="...",
            ground_truth=sample.ground_truth,
            diagnostics=diagnostics,
        )
```

### 3. Three-Class Decision

The adapter automatically classifies decisions based on diagnostics:

- **POSITIVE**: Core elements explicitly satisfied AND coverage ≥ 0.7
- **UNCERTAIN**: Coverage 0.4-0.7 OR core elements functionally matched
- **NEGATIVE**: Coverage < 0.4 OR core elements missing

The classification affects bullet scoring:
- POSITIVE → increment helpful for used bullets
- NEGATIVE with coverage ≥ 0.5 → increment harmful (over-conservative)
- UNCERTAIN → increment neutral

### 4. Constrained Curation

Limit curator operations to prevent playbook bloat:

```python
from ace import Curator, CurationRules

# Configure curation rules
curation_rules = CurationRules(
    similarity_threshold=0.85,  # ADD only if similarity < 0.85
    max_add_per_iteration=2,    # Limit to 2 ADD operations
)

# Create curator with validation
curator = Curator(
    llm_client,
    enable_validation=True,
    curation_rules=curation_rules,
)
```

Operations are prioritized:
1. UPDATE - preferred for refining existing bullets
2. TAG - for metadata updates
3. DEPRECATE/REMOVE - for harmful bullets
4. ADD - restricted by similarity check and count limit

### 5. Performance Reporting

Generate reports after training:

```python
from ace import PlaybookReporter

# Create reporter
reporter = PlaybookReporter(playbook)

# Get top positive contributors
top_positive = reporter.top_positive_contributors(top_n=10)
for bullet_report in top_positive:
    print(f"{bullet_report.id}: helpful={bullet_report.helpful}, "
          f"harmful={bullet_report.harmful}, score={bullet_report.score:.3f}")

# Get top negative contributors
top_negative = reporter.top_negative_contributors(top_n=10)

# Get deprecation candidates
candidates = reporter.deprecation_candidates(
    min_harmful_ratio=0.6,  # 60% harmful
    min_total_uses=3,       # At least 3 uses
)

# Export reports
reporter.export_markdown("reports/playbook_analysis.md", top_n=10)
reporter.export_csv("reports/playbook_analysis", top_n=10)
```

## Complete Example

```python
from ace import (
    OfflineAdapter, Generator, Reflector, Curator,
    Playbook, Sample, TaskEnvironment, EnvironmentResult,
    Diagnostics, ElementExtractor, PlaybookReporter,
)
from your_llm_module import YourLLMClient

# Initialize
llm = YourLLMClient(api_key="...")
playbook = Playbook()
extractor = ElementExtractor()

# Create environment with diagnostics
class PatentEnvironment(TaskEnvironment):
    def evaluate(self, sample, generator_output):
        # Your evaluation logic
        elements = generator_output.raw.get("elements", [])
        
        # Match elements
        matches = []
        for i, elem in enumerate(elements):
            match = extractor.match_element(elem, sample.context, is_core=True)
            match.id = f"E{i+1}"
            matches.append(match)
        
        coverage = extractor.compute_coverage(matches)
        
        diagnostics = Diagnostics(
            elements=matches,
            coverage=coverage,
            decision_basis_bullets=generator_output.bullet_ids,
        )
        
        return EnvironmentResult(
            feedback=self._compute_feedback(sample, generator_output),
            ground_truth=sample.ground_truth,
            diagnostics=diagnostics,
        )
    
    def _compute_feedback(self, sample, output):
        # Your feedback logic
        pass

# Create adapter with all enhancements
adapter = OfflineAdapter(
    playbook=playbook,
    generator=Generator(llm),
    reflector=Reflector(llm),
    curator=Curator(llm, enable_validation=True),
    enable_gating=True,
    reflection_window=20,
)

# Train
samples = [Sample(question="...", context="...", ground_truth="...")]
results = adapter.run(samples, PatentEnvironment(), epochs=3)

# Generate reports
reporter = PlaybookReporter(playbook)
reporter.export_markdown("playbook_report.md")

# Review and merge
# Manually review deprecation candidates and merge similar bullets
```

## Configuration Parameters

### Default Values

- `K (top_k)`: 25
- `guard_strategies`: 5
- `reflection_window`: 20
- `similarity_threshold` (gating): 0.1
- `similarity_threshold` (ADD validation): 0.85
- `coverage_thresholds`: POS≥0.7, UNC 0.4-0.7, NEG<0.4
- `core_weight`: 3.0
- `non_core_weight`: 1.0
- `max_add_per_iteration`: 2

### Customization

All parameters can be customized via configuration objects:

```python
from ace import GatingConfig, CurationRules

gating_config = GatingConfig(
    top_k=30,
    guard_strategies=3,
    similarity_threshold=0.15,
)

curation_rules = CurationRules(
    similarity_threshold=0.80,
    max_add_per_iteration=3,
)
```

## Best Practices

1. **Enable gating gradually**: Start with `enable_gating=False`, then enable after initial training
2. **Monitor reports**: Review reports every few epochs to identify problematic bullets
3. **Manual review**: Regularly review deprecation candidates before removing
4. **Adjust thresholds**: Tune coverage thresholds based on your task's precision/recall trade-off
5. **Use guard strategies**: Define 3-5 core bullets that should always be included
6. **Limit ADD operations**: Keep `max_add_per_iteration` low (1-2) to prevent bloat
7. **Prioritize UPDATE**: Refine existing bullets rather than adding new ones

## Migration from Previous Version

The new features are backward compatible. To migrate:

1. No changes needed for basic usage (all enhancements are opt-in)
2. To enable enhancements, set flags when creating adapter:
   ```python
   adapter = OfflineAdapter(
       ...,
       enable_gating=True,  # Add this
       reflection_window=20,  # Update this (was 3)
   )
   ```
3. Update your TaskEnvironment to return diagnostics (optional but recommended)
4. Enable curator validation:
   ```python
   curator = Curator(llm, enable_validation=True)
   ```

## Troubleshooting

### Issue: Gating is slow
- Solution: Disable gating during initial development (`enable_gating=False`)
- Or: Use cached embeddings (not yet implemented)

### Issue: Too many ADD operations
- Solution: Reduce `max_add_per_iteration` or increase `similarity_threshold`

### Issue: Bullets not being deprecated
- Solution: Review `min_harmful_ratio` and `min_total_uses` in reporting

### Issue: Coverage always low
- Solution: Check element extraction and matching logic; adjust core_weight/non_core_weight
