"""Curation rules for validating and constraining playbook operations."""

from __future__ import annotations

from typing import List, Optional

from .delta import DeltaOperation
from .gating import BulletGate
from .playbook import Playbook


class CurationRules:
    """Validates and constrains curator operations."""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_add_per_iteration: int = 2,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_add_per_iteration = max_add_per_iteration
        self.bullet_gate = BulletGate()
    
    def validate_operations(
        self,
        operations: List[DeltaOperation],
        playbook: Playbook,
    ) -> List[DeltaOperation]:
        """
        Validate and filter operations according to curation rules.
        
        Rules:
        1. Prioritize UPDATE over ADD
        2. Limit ADD operations to max_add_per_iteration
        3. For ADD operations, check similarity with existing bullets
        4. Only allow ADD if similarity < similarity_threshold
        5. Support UPDATE, TAG, DEPRECATE (REMOVE), ADD
        
        Args:
            operations: Proposed operations from curator
            playbook: Current playbook state
            
        Returns:
            Filtered list of valid operations
        """
        validated = []
        add_count = 0
        
        # First pass: prioritize UPDATE, TAG, and DEPRECATE operations
        for op in operations:
            op_type = op.type.upper()
            
            if op_type == "UPDATE":
                # Allow UPDATE if bullet exists
                if op.bullet_id and playbook.get_bullet(op.bullet_id):
                    validated.append(op)
            
            elif op_type == "TAG":
                # Allow TAG if bullet exists
                if op.bullet_id and playbook.get_bullet(op.bullet_id):
                    validated.append(op)
            
            elif op_type in ("REMOVE", "DEPRECATE"):
                # Allow REMOVE/DEPRECATE if bullet exists
                # Rename DEPRECATE to REMOVE for consistency
                if op.bullet_id and playbook.get_bullet(op.bullet_id):
                    validated.append(
                        DeltaOperation(
                            type="REMOVE",
                            section=op.section,
                            bullet_id=op.bullet_id,
                            metadata=op.metadata,
                        )
                    )
        
        # Second pass: process ADD operations with constraints
        for op in operations:
            if op.type.upper() != "ADD":
                continue
            
            # Check ADD limit
            if add_count >= self.max_add_per_iteration:
                break
            
            # Check if content is similar to existing bullets
            if self._is_too_similar(op.content or "", playbook):
                continue
            
            # Allow this ADD operation
            validated.append(op)
            add_count += 1
        
        return validated
    
    def _is_too_similar(self, new_content: str, playbook: Playbook) -> bool:
        """
        Check if new content is too similar to existing bullets.
        
        Returns:
            True if similarity >= threshold with any existing bullet
        """
        if not new_content:
            return False
        
        for bullet in playbook.bullets():
            similarity = self.bullet_gate.compute_similarity(new_content, bullet.content)
            if similarity >= self.similarity_threshold:
                return True
        
        return False
    
    def should_deprecate(
        self,
        bullet_id: str,
        playbook: Playbook,
        min_harmful_ratio: float = 0.6,
        min_total_uses: int = 3,
    ) -> bool:
        """
        Determine if a bullet should be deprecated.
        
        Criteria:
        - harmful / (helpful + harmful) >= min_harmful_ratio
        - helpful + harmful >= min_total_uses
        
        Args:
            bullet_id: Bullet to check
            playbook: Current playbook
            min_harmful_ratio: Minimum ratio of harmful to total
            min_total_uses: Minimum total uses before considering deprecation
            
        Returns:
            True if bullet should be deprecated
        """
        bullet = playbook.get_bullet(bullet_id)
        if not bullet:
            return False
        
        total_uses = bullet.helpful + bullet.harmful
        if total_uses < min_total_uses:
            return False
        
        harmful_ratio = bullet.harmful / total_uses if total_uses > 0 else 0
        return harmful_ratio >= min_harmful_ratio
