"""Task-specific prompts for patent matching classification task."""

GENERATOR_PROMPT_PATENT_CLS = """You are an expert patent analyzer performing candidate classification.

PLAYBOOK (previous learnings):
{playbook}

PREVIOUS REFLECTIONS:
{reflection}

TASK:
Given a patent question and candidate contexts, classify each candidate as either "positive" or "negative".
- positive: The candidate directly addresses the patent question with relevant claims or technical details
- negative: The candidate does not match the question or provides irrelevant information

QUESTION:
{question}

CONTEXT (candidates to classify):
{context}

INSTRUCTIONS:
1. For each candidate, break down the patent claim into key technical elements
2. Check if the candidate provides evidence for EACH element (not just surface keyword matches)
3. Avoid being fooled by similar terminology without substantive alignment
4. Consider element-wise evidence mapping before making final judgment

STRICT OUTPUT (valid JSON only, NO code fences, NO extra text):
{{
  "reasoning": "Step-by-step analysis of classification strategy and element alignment",
  "final_answer": "Summary of classification results, e.g., 'Selected positive IDs: [id1, id2]'",
  "bullet_ids": ["list", "of", "relevant", "bullet", "ids", "from", "playbook"],
  "predictions": [
    {{
      "id": "candidate_id",
      "label": "positive|negative",
      "reason": "One sentence explanation of why this classification was made"
    }}
  ]
}}"""

REFLECTOR_PROMPT_PATENT_CLS = """You are a reflection expert analyzing classification errors in patent matching.

QUESTION:
{question}

YOUR REASONING:
{reasoning}

YOUR PREDICTION:
{prediction}

GROUND TRUTH:
{ground_truth}

ENVIRONMENT FEEDBACK (includes metrics and error analysis):
{feedback}

PLAYBOOK BULLETS YOU USED:
{playbook_excerpt}

TASK:
Analyze the classification errors deeply:
1. Parse the feedback JSON to identify false positives (FP) and false negatives (FN)
2. For each error type (FP from negative, FP from hard_negative, FN from positive):
   - Identify the root cause (e.g., keyword distraction, missing element verification, incomplete reasoning)
   - Determine what correct approach should have been used
3. Extract ONE key actionable insight that would prevent similar errors
4. Tag relevant playbook bullets with status: "keep", "revise", or "remove"

STRICT OUTPUT (valid JSON only, NO code fences, NO extra text):
{{
  "reasoning": "Analysis of what went wrong in the classification process",
  "error_identification": "Detailed breakdown of FP and FN errors by type with examples",
  "root_cause_analysis": "Deep analysis of why these errors occurred (e.g., element verification gaps, keyword traps)",
  "correct_approach": "Specific strategy that should have been used to avoid these errors",
  "key_insight": "ONE concise, actionable principle for better classification (e.g., 'Verify ALL claim elements before labeling positive')",
  "bullet_tags": [
    {{"id": "bullet_id", "tag": "keep|revise|remove"}}
  ]
}}"""

CURATOR_PROMPT_PATENT_CLS = """You are a playbook curator converting reflection insights into concrete classification guidelines.

CURRENT PLAYBOOK:
{playbook}

PLAYBOOK STATS:
{stats}

PROGRESS:
{progress}

REFLECTION (from error analysis):
{reflection}

QUESTION CONTEXT:
{question_context}

TASK:
Transform the reflection's key_insight and error analysis into playbook operations:
1. ADD new bullets for:
   - Element verification checklists
   - Common trap patterns to avoid (keyword distractions, surface similarities)
   - Evidence alignment requirements
2. UPDATE existing bullets if they need refinement based on new insights
3. TAG (via metadata counters) bullets that led to errors:
   - Use {{"helpful": 1}} to upvote helpful bullets
   - Use {{"harmful": 1}} to downvote harmful bullets
4. REMOVE clearly harmful or redundant bullets

Constraints:
- Use section "defaults" for new bullets
- Keep bullets concise and actionable (3-5 operations per iteration)
- Preserve interpretability and avoid redundancy

STRICT OUTPUT (valid JSON only, NO code fences, NO extra text):
{{
  "reasoning": "Brief explanation of how the operations reflect the reflection insights",
  "operations": [
    {{
      "type": "ADD",
      "section": "defaults",
      "content": "Concise guideline focused on element-wise verification or trap avoidance",
      "metadata": {{"helpful": 1}}
    }},
    {{
      "type": "UPDATE",
      "bullet_id": "existing_bullet_id",
      "content": "Refined guideline incorporating new insight"
    }},
    {{
      "type": "TAG",
      "bullet_id": "existing_bullet_id",
      "metadata": {{"helpful": 1}}
    }},
    {{
      "type": "REMOVE",
      "bullet_id": "bullet_id_to_remove"
    }}
  ]
}}"""