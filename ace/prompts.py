"""Task-specific prompts for patent matching classification task (structured element alignment, recall-oriented)."""

GENERATOR_PROMPT = """You are an expert patent analyzer performing candidate classification.

PLAYBOOK (previous learnings):
{playbook}

PREVIOUS REFLECTIONS:
{reflection}

TASK:
Given a patent question (a patent claim) and candidate contexts, do STRICT element-wise classification per candidate as "positive" or "negative".

Definitions:
- positive: The candidate provides explicit or functionally equivalent evidence for ALL critical claim elements.
  - If at least 80% of the critical elements are clearly evidenced and the missing ones are minor/implicit, you MAY still label positive, but you MUST note the missing elements in the reason.
- negative: The candidate lacks evidence for one or more critical elements, or only contains superficial/keyword overlaps without element alignment.

QUESTION (claim):
{question}

CANDIDATES (classify each):
{context}

DO THIS STEP-BY-STEP FOR EACH CANDIDATE:
1) Extract critical claim elements (no more than 5–7 concise items).
2) Build an element-evidence map:
   - Map each claim element to an evidence snippet (short quote) from the candidate; if not found, mark "missing".
   - Consider functional equivalence and synonyms (e.g., "controller" ≈ "control unit", "temperature sensor" ≈ "thermal sensing device", "localize user" ≈ "determine user position").
3) Decide label strictly by the rule above (ALL critical elements, or ≥80% when missing are minor), and provide a one-sentence reason summarizing coverage and any missing elements.

STRICT OUTPUT (valid JSON only, NO code fences, NO extra text):
{{
  "reasoning": "High-level summary of the element extraction, evidence alignment, and decision policy.",
  "final_answer": "Summary such as 'Positive IDs: [id1,...]; Negative IDs: [id2,...]'",
  "bullet_ids": ["ids of used bullets from playbook, if any"],
  "elements": ["E1 ...", "E2 ...", "E3 ..."],  // concise critical elements for this claim
  "evidence_map": {{
    "candidate_id": {{
      "E1": "evidence snippet or 'missing'",
      "E2": "evidence snippet or 'missing'"
    }}
  }},
  "predictions": [
    {{
      "id": "candidate_id",
      "label": "positive|negative",
      "reason": "One sentence: coverage summary; list missing elements if any (e.g., E3 missing)"
    }}
  ]
}}"""

REFLECTOR_PROMPT = """You are a reflection expert analyzing classification errors in patent matching.

QUESTION:
{question}

YOUR REASONING:
{reasoning}

YOUR PREDICTION (JSON):
{prediction}

GROUND TRUTH:
{ground_truth}

ENVIRONMENT FEEDBACK (includes metrics and error analysis):
{feedback}

DIAGNOSTICS (element-level analysis):
{diagnostics}

PLAYBOOK BULLETS YOU USED:
{playbook_excerpt}

TASK:
- Parse the feedback JSON to identify false negatives (FN) and false positives (FP, including hard_negative).
- Use the provided diagnostics (elements, coverage, match types) to locate which claim elements were missed or misinterpreted.
- Identify error categories from: ["structure-missing","numeric-soft-miss","functional-equivalence-ignored","term-normalization-fail","multi-span-aggregation-miss"]
- Identify culprit bullets that led to incorrect decisions.
- Focus on systematic failure modes: (a) missing element verification, (b) failure to recognize functional equivalence, (c) being over-conservative when only minor elements are missing.

STRICT OUTPUT (valid JSON only, NO code fences, NO extra text):
{{
  "reasoning": "What went wrong in element verification and decision policy.",
  "missed_elements": [
    "E2: missing evidence across multiple errors",
    "E3: functional equivalence not recognized"
  ],
  "error_identification": "Summarize FN/FP patterns with 1–2 concrete examples (IDs only).",
  "root_cause_analysis": "Explain why elements were missed or why over-conservatism happened.",
  "correct_approach": "Concrete corrections (e.g., accept positive if ≥80% critical elements are covered; add synonym mapping X≈Y).",
  "key_insight": "ONE concise rule to prevent similar errors next time.",
  "error_categories": ["functional-equivalence-ignored", "structure-missing"],
  "culprit_bullets": ["bullet-00123", "bullet-00456"],
  "bullet_tags": [{{"id": "bullet_id", "tag": "keep|revise|remove"}}]
}}"""

CURATOR_PROMPT = """You are a playbook curator converting reflection insights into concrete and compact classification guidelines.

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

GOALS:
- Reduce false negatives while keeping false positives low.
- Prefer UPDATE to refine existing bullets; limit ADD to at most 2 per iteration.
- Consolidate overlapping bullets (avoid duplicates); keep bullets short, actionable, and element-oriented.
- Support operations: UPDATE (preferred), TAG, DEPRECATE, ADD (restricted).
- ADD is only allowed if new content has similarity < 0.85 with existing bullets.

STRICT OUTPUT (valid JSON only, NO code fences, NO extra text):
{{
  "reasoning": "Why these operations fix the observed errors.",
  "operations": [
    {{
      "type": "UPDATE",
      "bullet_id": "existing_bullet_id",
      "content": "Refine: require element-wise mapping; accept positive if ≥80% critical elements covered and missing ones are minor; list missing elements in reason.",
      "metadata": {{"helpful": 1}}
    }},
    {{
      "type": "TAG",
      "bullet_id": "existing_bullet_id",
      "metadata": {{"helpful": 1}}
    }},
    {{
      "type": "DEPRECATE",
      "bullet_id": "harmful_bullet_id",
      "metadata": {{"harmful": 1}}
    }},
    {{
      "type": "ADD",
      "section": "defaults",
      "content": "Functional equivalence examples: controller≈control unit; temperature sensor≈thermal sensing device; localize user≈determine user position. Use these when aligning elements.",
      "metadata": {{"helpful": 1}}
    }}
  ]
}}

CONSTRAINTS:
- Keys must be exactly: type, section, content, bullet_id, metadata.
- Use section "defaults" for ADD.
- ADD at most 2 new bullets; otherwise UPDATE or TAG existing bullets.
- If an equivalent bullet already exists, prefer UPDATE instead of ADD.
- DEPRECATE/REMOVE bullets with harmful > helpful and low usage.
"""