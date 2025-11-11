"""Bullet gating for selective context injection using similarity-based selection."""

from __future__ import annotations

from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class GatingConfig:
    """Configuration for bullet gating."""
    
    top_k: int = 25
    guard_strategies: int = 5
    similarity_threshold: float = 0.1  # Minimum similarity to consider
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class BulletGate:
    """Selects relevant bullets based on sample similarity."""
    
    def __init__(self, config: Optional[GatingConfig] = None, auto_load_encoder: bool = False):
        self.config = config or GatingConfig()
        self._encoder: Optional[object] = None
        
        if EMBEDDINGS_AVAILABLE and auto_load_encoder:
            try:
                self._encoder = SentenceTransformer(self.config.model_name)
            except Exception:
                # If model download fails, gating will fall back to returning all bullets
                self._encoder = None
    
    def select_bullets(
        self,
        sample_text: str,
        bullets: List[Tuple[str, str]],  # List of (bullet_id, content)
        guard_bullets: Optional[List[Tuple[str, str]]] = None,
    ) -> List[str]:
        """
        Select Top-K most relevant bullets plus guard strategies.
        
        Args:
            sample_text: The question/context to match against
            bullets: List of (bullet_id, content) tuples
            guard_bullets: Optional list of guard strategy bullets to always include
            
        Returns:
            List of selected bullet IDs
        """
        if not EMBEDDINGS_AVAILABLE or self._encoder is None:
            # Fallback: return all bullet IDs
            return [bid for bid, _ in bullets]
        
        if not bullets:
            return []
        
        # Always include guard bullets
        selected_ids = []
        guard_ids = set()
        if guard_bullets:
            guard_ids = {bid for bid, _ in guard_bullets[:self.config.guard_strategies]}
            selected_ids.extend(guard_ids)
        
        # Filter out guard bullets from main selection
        non_guard_bullets = [(bid, content) for bid, content in bullets if bid not in guard_ids]
        
        if not non_guard_bullets:
            return selected_ids
        
        try:
            # Encode sample and bullet contents
            sample_embedding = self._encoder.encode([sample_text])
            bullet_contents = [content for _, content in non_guard_bullets]
            bullet_embeddings = self._encoder.encode(bullet_contents)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(sample_embedding, bullet_embeddings)[0]
            
            # Get top-K indices by similarity
            num_to_select = min(self.config.top_k, len(non_guard_bullets))
            top_indices = np.argsort(similarities)[::-1][:num_to_select]
            
            # Filter by threshold and add to selection
            for idx in top_indices:
                if similarities[idx] >= self.config.similarity_threshold:
                    selected_ids.append(non_guard_bullets[idx][0])
            
        except Exception:
            # On any error, fall back to returning all bullets
            selected_ids.extend([bid for bid, _ in non_guard_bullets])
        
        return selected_ids
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Returns:
            Similarity score in [0, 1] range, or 0.0 if embeddings unavailable.
        """
        if not EMBEDDINGS_AVAILABLE or self._encoder is None:
            return 0.0
        
        try:
            embeddings = self._encoder.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception:
            return 0.0
