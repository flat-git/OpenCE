"""Diagnostics module for element-level analysis and tracking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


MatchType = Literal["explicit", "functional", "none"]


@dataclass
class ElementMatch:
    """Represents a single element and its match status."""
    
    id: str  # Element identifier (e.g., "E1", "E2")
    canonical: str  # Canonical description of the element
    is_core: bool  # Whether this is a core element
    match: MatchType  # Type of match found
    confidence: float  # Confidence in the match [0, 1]
    evidence: Optional[str] = None  # Evidence snippet if matched


@dataclass
class Diagnostics:
    """Element-level diagnostics for a prediction."""
    
    elements: List[ElementMatch] = field(default_factory=list)
    coverage: float = 0.0  # Overall coverage score [0, 1]
    decision_basis_bullets: List[str] = field(default_factory=list)  # Bullet IDs used
    
    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "elements": [
                {
                    "id": e.id,
                    "canonical": e.canonical,
                    "is_core": e.is_core,
                    "match": e.match,
                    "confidence": e.confidence,
                    "evidence": e.evidence,
                }
                for e in self.elements
            ],
            "coverage": self.coverage,
            "decision_basis_bullets": self.decision_basis_bullets,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Diagnostics":
        """Create from dictionary."""
        elements_data = data.get("elements", [])
        elements = []
        if isinstance(elements_data, list):
            for elem in elements_data:
                if isinstance(elem, dict):
                    elements.append(
                        ElementMatch(
                            id=str(elem.get("id", "")),
                            canonical=str(elem.get("canonical", "")),
                            is_core=bool(elem.get("is_core", False)),
                            match=elem.get("match", "none"),  # type: ignore
                            confidence=float(elem.get("confidence", 0.0)),
                            evidence=elem.get("evidence"),
                        )
                    )
        
        return cls(
            elements=elements,
            coverage=float(data.get("coverage", 0.0)),
            decision_basis_bullets=[
                str(bid) for bid in data.get("decision_basis_bullets", [])
            ],
        )


class ElementExtractor:
    """Extracts and matches elements from generator outputs."""
    
    # Common synonyms for functional equivalence
    SYNONYMS = {
        "controller": ["control unit", "control module", "processor"],
        "temperature sensor": ["thermal sensor", "thermal sensing device", "temperature detector"],
        "localize": ["determine position", "determine location", "identify location"],
        "user": ["person", "individual", "operator"],
        "device": ["apparatus", "equipment", "unit", "system"],
        "method": ["process", "procedure", "technique"],
        "system": ["apparatus", "device", "configuration"],
    }
    
    def extract_elements(
        self,
        generator_output_elements: Optional[List[str]] = None,
        question: Optional[str] = None,
    ) -> List[str]:
        """
        Extract critical elements from generator output or question.
        
        Args:
            generator_output_elements: Elements from generator's structured output
            question: Question text to extract from if generator output not available
            
        Returns:
            List of element descriptions
        """
        if generator_output_elements:
            return generator_output_elements
        
        if question:
            # Simple rule-based extraction
            # Look for phrases like "comprising", "including", "having"
            elements = []
            parts = re.split(r'\bcomprising\b|\bincluding\b|\bhaving\b|\bwherein\b', question, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Extract from claim body
                claim_body = parts[1]
                # Split by common separators
                items = re.split(r';|\band\b', claim_body, flags=re.IGNORECASE)
                for item in items[:7]:  # Limit to 7 elements
                    clean = item.strip().strip(',').strip()
                    if clean and len(clean) > 5:
                        elements.append(clean[:200])  # Limit length
            
            if not elements and question:
                # Fallback: extract noun phrases (very simple)
                elements.append(question[:200])
            
            return elements
        
        return []
    
    def match_element(
        self,
        element: str,
        evidence_text: str,
        is_core: bool = True,
    ) -> ElementMatch:
        """
        Match a single element against evidence text.
        
        Args:
            element: The element to match
            evidence_text: Text to search for evidence
            is_core: Whether this is a core element
            
        Returns:
            ElementMatch with match type and confidence
        """
        element_lower = element.lower()
        evidence_lower = evidence_text.lower()
        
        # Check for explicit match (direct substring)
        if element_lower in evidence_lower:
            return ElementMatch(
                id="",  # Will be set by caller
                canonical=element,
                is_core=is_core,
                match="explicit",
                confidence=1.0,
                evidence=self._extract_snippet(evidence_text, element),
            )
        
        # Check for functional equivalence using synonyms
        for base_term, synonyms in self.SYNONYMS.items():
            if base_term in element_lower:
                for synonym in synonyms:
                    if synonym in evidence_lower:
                        return ElementMatch(
                            id="",
                            canonical=element,
                            is_core=is_core,
                            match="functional",
                            confidence=0.7,
                            evidence=self._extract_snippet(evidence_text, synonym),
                        )
        
        # No match found
        return ElementMatch(
            id="",
            canonical=element,
            is_core=is_core,
            match="none",
            confidence=0.0,
            evidence=None,
        )
    
    def _extract_snippet(self, text: str, term: str, context_chars: int = 100) -> str:
        """Extract a snippet around the matching term."""
        text_lower = text.lower()
        term_lower = term.lower()
        idx = text_lower.find(term_lower)
        if idx == -1:
            return text[:context_chars]
        
        start = max(0, idx - context_chars // 2)
        end = min(len(text), idx + len(term) + context_chars // 2)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet
    
    def compute_coverage(
        self,
        elements: List[ElementMatch],
        core_weight: float = 3.0,
        non_core_weight: float = 1.0,
    ) -> float:
        """
        Compute overall coverage score.
        
        Weighted average of element match confidences.
        Core elements have higher weight.
        
        Returns:
            Coverage score in [0, 1]
        """
        if not elements:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for elem in elements:
            weight = core_weight if elem.is_core else non_core_weight
            total_weight += weight
            weighted_sum += elem.confidence * weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
