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

OUTPUT FORMAT (strict JSON only):
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
}}

IMPORTANT:
- You MUST output valid JSON only, no extra text
- Every candidate in context must have a prediction
- Label must be exactly "positive" or "negative"
- Reason should explain element-wise evidence alignment
"""

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

OUTPUT FORMAT (strict JSON only):
{{
  "reasoning": "Analysis of what went wrong in the classification process",
  "error_identification": "Detailed breakdown of FP and FN errors by type with examples",
  "root_cause_analysis": "Deep analysis of why these errors occurred (e.g., element verification gaps, keyword traps)",
  "correct_approach": "Specific strategy that should have been used to avoid these errors",
  "key_insight": "ONE concise, actionable principle for better classification (e.g., 'Verify ALL claim elements before labeling positive')",
  "bullet_tags": [
    {{"id": "bullet_id", "tag": "keep|revise|remove"}}
  ]
}}

IMPORTANT:
- Output valid JSON only
- Focus on systematic error patterns, not individual mistakes
- Key insight must be concise and directly actionable
"""

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
3. REMOVE or TAG bullets that led to errors

Focus on classification strategy improvements:
- How to decompose patent claims into verifiable elements
- How to avoid false positive traps (negative/hard_negative mislabeled as positive)
- How to catch false negatives (missed positive cases)
- Specific verification steps before assigning labels

OUTPUT FORMAT (strict JSON only):
{{
  "operations": [
    {{
      "operation": "ADD",
      "content": "Concise guideline focused on element-wise verification or trap avoidance",
      "tags": ["classification", "element_check"]
    }},
    {{
      "operation": "UPDATE",
      "bullet_id": "existing_bullet_id",
      "content": "Refined guideline incorporating new insight"
    }},
    {{
      "operation": "TAG",
      "bullet_id": "existing_bullet_id",
      "tag": "keep|harmful|outdated"
    }},
    {{
      "operation": "REMOVE",
      "bullet_id": "bullet_id_to_remove"
    }}
  ]
}}

IMPORTANT:
- Output valid JSON only, no extra text
- Each bullet should be actionable and specific to patent classification
- Focus on preventing the error patterns identified in reflection
- Limit to 3-5 operations per iteration to maintain playbook quality
"""
