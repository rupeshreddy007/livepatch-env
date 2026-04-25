"""Curriculum controller — tracks mastery per fault type and escalates difficulty."""
from typing import Dict, List, Optional
from .config import ALL_FAULT_TYPES, ALL_DIFFICULTIES


class CurriculumController:
    """Tracks agent mastery per fault type and selects difficulty."""

    def __init__(self):
        self.episode_count = 0
        self.fault_attempts: Dict[str, int] = {f: 0 for f in ALL_FAULT_TYPES}
        self.fault_successes: Dict[str, int] = {f: 0 for f in ALL_FAULT_TYPES}
        self.recent_scores: List[float] = []
        self.current_difficulty_idx = 0

    @property
    def current_difficulty(self) -> str:
        return ALL_DIFFICULTIES[min(self.current_difficulty_idx, len(ALL_DIFFICULTIES) - 1)]

    def mastery(self, fault_type: str) -> float:
        """Resolution rate for a fault type."""
        attempts = self.fault_attempts.get(fault_type, 0)
        if attempts == 0:
            return 0.0
        return self.fault_successes.get(fault_type, 0) / attempts

    def weakest_faults(self, n: int = 3) -> List[str]:
        """Return the N fault types the agent struggles with most."""
        scored = [(f, self.mastery(f)) for f in ALL_FAULT_TYPES]
        scored.sort(key=lambda x: x[1])
        return [f for f, _ in scored[:n]]

    def record_episode(self, faults_injected: List[str],
                        faults_resolved: List[str], score: float):
        """Record results of an episode."""
        self.episode_count += 1
        self.recent_scores.append(score)
        if len(self.recent_scores) > 10:
            self.recent_scores.pop(0)

        for f in faults_injected:
            self.fault_attempts[f] = self.fault_attempts.get(f, 0) + 1
        for f in faults_resolved:
            self.fault_successes[f] = self.fault_successes.get(f, 0) + 1

        # Check if we should escalate difficulty
        if len(self.recent_scores) >= 5:
            avg = sum(self.recent_scores[-5:]) / 5
            if avg > 0.7 and self.current_difficulty_idx < len(ALL_DIFFICULTIES) - 1:
                self.current_difficulty_idx += 1
            elif avg < 0.3 and self.current_difficulty_idx > 0:
                self.current_difficulty_idx -= 1

    def summary(self) -> Dict:
        """Return curriculum state summary."""
        return {
            "episode_count": self.episode_count,
            "current_difficulty": self.current_difficulty,
            "mastery_by_fault": {f: round(self.mastery(f), 3) for f in ALL_FAULT_TYPES},
            "weakest_faults": self.weakest_faults(3),
            "recent_avg_score": round(
                sum(self.recent_scores) / max(1, len(self.recent_scores)), 3
            ) if self.recent_scores else 0.0,
        }
