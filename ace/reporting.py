"""Reporting module for playbook analysis and bullet performance tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .playbook import Bullet, Playbook


@dataclass
class BulletReport:
    """Report entry for a single bullet."""
    
    id: str
    section: str
    content: str
    helpful: int
    harmful: int
    neutral: int
    score: float  # (helpful - harmful) * relevance
    tags: List[str]


class PlaybookReporter:
    """Generates reports on playbook bullet performance."""
    
    def __init__(self, playbook: Playbook):
        self.playbook = playbook
    
    def top_positive_contributors(self, top_n: int = 10) -> List[BulletReport]:
        """
        Get bullets with highest positive contribution (high helpful, low harmful).
        
        Sorted by: (helpful - harmful), then by helpful count.
        """
        bullets = self.playbook.bullets()
        
        # Calculate contribution score
        scored = []
        for bullet in bullets:
            contribution = bullet.helpful - bullet.harmful
            scored.append((bullet, contribution))
        
        # Sort by contribution (desc), then helpful (desc)
        scored.sort(key=lambda x: (x[1], x[0].helpful), reverse=True)
        
        # Convert to reports
        reports = []
        for bullet, contribution in scored[:top_n]:
            reports.append(
                BulletReport(
                    id=bullet.id,
                    section=bullet.section,
                    content=bullet.content[:100] + ("..." if len(bullet.content) > 100 else ""),
                    helpful=bullet.helpful,
                    harmful=bullet.harmful,
                    neutral=bullet.neutral,
                    score=bullet.relevance_score,
                    tags=bullet.tags,
                )
            )
        
        return reports
    
    def top_negative_contributors(self, top_n: int = 10) -> List[BulletReport]:
        """
        Get bullets with highest negative contribution (high harmful, low helpful).
        
        Sorted by: (harmful - helpful), then by harmful count.
        """
        bullets = self.playbook.bullets()
        
        # Calculate negative contribution
        scored = []
        for bullet in bullets:
            negative_contribution = bullet.harmful - bullet.helpful
            if negative_contribution > 0:  # Only include bullets with net negative
                scored.append((bullet, negative_contribution))
        
        # Sort by negative contribution (desc), then harmful (desc)
        scored.sort(key=lambda x: (x[1], x[0].harmful), reverse=True)
        
        # Convert to reports
        reports = []
        for bullet, _ in scored[:top_n]:
            reports.append(
                BulletReport(
                    id=bullet.id,
                    section=bullet.section,
                    content=bullet.content[:100] + ("..." if len(bullet.content) > 100 else ""),
                    helpful=bullet.helpful,
                    harmful=bullet.harmful,
                    neutral=bullet.neutral,
                    score=bullet.relevance_score,
                    tags=bullet.tags,
                )
            )
        
        return reports
    
    def deprecation_candidates(
        self,
        min_harmful_ratio: float = 0.6,
        min_total_uses: int = 3,
    ) -> List[BulletReport]:
        """
        Get bullets that are candidates for deprecation.
        
        Criteria:
        - harmful / (helpful + harmful) >= min_harmful_ratio
        - helpful + harmful >= min_total_uses
        
        Sorted by harmful ratio (desc).
        """
        bullets = self.playbook.bullets()
        
        candidates = []
        for bullet in bullets:
            total_uses = bullet.helpful + bullet.harmful
            if total_uses < min_total_uses:
                continue
            
            harmful_ratio = bullet.harmful / total_uses
            if harmful_ratio >= min_harmful_ratio:
                candidates.append((bullet, harmful_ratio))
        
        # Sort by harmful ratio (desc)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to reports
        reports = []
        for bullet, harmful_ratio in candidates:
            reports.append(
                BulletReport(
                    id=bullet.id,
                    section=bullet.section,
                    content=bullet.content[:100] + ("..." if len(bullet.content) > 100 else ""),
                    helpful=bullet.helpful,
                    harmful=bullet.harmful,
                    neutral=bullet.neutral,
                    score=bullet.relevance_score,
                    tags=bullet.tags,
                )
            )
        
        return reports
    
    def export_markdown(
        self,
        filepath: str,
        top_n: int = 10,
        min_harmful_ratio: float = 0.6,
        min_total_uses: int = 3,
    ) -> None:
        """
        Export all three reports to a markdown file.
        
        Args:
            filepath: Path to save markdown file
            top_n: Number of bullets to include in top lists
            min_harmful_ratio: Minimum harmful ratio for deprecation
            min_total_uses: Minimum total uses for deprecation
        """
        positive = self.top_positive_contributors(top_n)
        negative = self.top_negative_contributors(top_n)
        deprecate = self.deprecation_candidates(min_harmful_ratio, min_total_uses)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Playbook Performance Report\n\n")
            
            # Top positive contributors
            f.write(f"## Top {top_n} Positive Contributors\n\n")
            f.write("| ID | Section | Helpful | Harmful | Neutral | Score | Content |\n")
            f.write("|----|---------|---------|---------|---------|-------|----------|\n")
            for report in positive:
                f.write(
                    f"| {report.id} | {report.section} | {report.helpful} | "
                    f"{report.harmful} | {report.neutral} | {report.score:.3f} | "
                    f"{report.content} |\n"
                )
            f.write("\n")
            
            # Top negative contributors
            f.write(f"## Top {top_n} Negative Contributors\n\n")
            f.write("| ID | Section | Helpful | Harmful | Neutral | Score | Content |\n")
            f.write("|----|---------|---------|---------|---------|-------|----------|\n")
            for report in negative:
                f.write(
                    f"| {report.id} | {report.section} | {report.helpful} | "
                    f"{report.harmful} | {report.neutral} | {report.score:.3f} | "
                    f"{report.content} |\n"
                )
            f.write("\n")
            
            # Deprecation candidates
            f.write(f"## Deprecation Candidates (harmful_ratio >= {min_harmful_ratio}, uses >= {min_total_uses})\n\n")
            f.write("| ID | Section | Helpful | Harmful | Neutral | Harmful Ratio | Content |\n")
            f.write("|----|---------|---------|---------|---------|---------------|----------|\n")
            for report in deprecate:
                total = report.helpful + report.harmful
                ratio = report.harmful / total if total > 0 else 0
                f.write(
                    f"| {report.id} | {report.section} | {report.helpful} | "
                    f"{report.harmful} | {report.neutral} | {ratio:.2%} | "
                    f"{report.content} |\n"
                )
            f.write("\n")
    
    def export_csv(
        self,
        filepath: str,
        top_n: int = 10,
        min_harmful_ratio: float = 0.6,
        min_total_uses: int = 3,
    ) -> None:
        """
        Export all three reports to CSV files.
        
        Creates three files:
        - {filepath}_positive.csv
        - {filepath}_negative.csv
        - {filepath}_deprecate.csv
        """
        import csv
        
        positive = self.top_positive_contributors(top_n)
        negative = self.top_negative_contributors(top_n)
        deprecate = self.deprecation_candidates(min_harmful_ratio, min_total_uses)
        
        # Export positive contributors
        with open(f"{filepath}_positive.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Section", "Helpful", "Harmful", "Neutral", "Score", "Tags", "Content"])
            for report in positive:
                writer.writerow([
                    report.id, report.section, report.helpful, report.harmful,
                    report.neutral, f"{report.score:.3f}", ",".join(report.tags), report.content
                ])
        
        # Export negative contributors
        with open(f"{filepath}_negative.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Section", "Helpful", "Harmful", "Neutral", "Score", "Tags", "Content"])
            for report in negative:
                writer.writerow([
                    report.id, report.section, report.helpful, report.harmful,
                    report.neutral, f"{report.score:.3f}", ",".join(report.tags), report.content
                ])
        
        # Export deprecation candidates
        with open(f"{filepath}_deprecate.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Section", "Helpful", "Harmful", "Neutral", "Harmful_Ratio", "Tags", "Content"])
            for report in deprecate:
                total = report.helpful + report.harmful
                ratio = report.harmful / total if total > 0 else 0
                writer.writerow([
                    report.id, report.section, report.helpful, report.harmful,
                    report.neutral, f"{ratio:.2%}", ",".join(report.tags), report.content
                ])
