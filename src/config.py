"""
Configuration management for network bootstrap analysis.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AnalysisConfig:
    """Configuration for network bootstrap analysis."""
    total_runs: int = 64
    inner_group_size: int = 8
    support_threshold: float = 0.8
    fdr_threshold: float = 0.05
    epsilon: float = 1e-10
    
    def is_valid(self) -> bool:
        """Validate configuration parameters."""
        if self.total_runs <= 0 or self.inner_group_size <= 0:
            return False
        if self.total_runs % self.inner_group_size != 0:
            return False
        if not 0 < self.support_threshold <= 1:
            return False
        if not 0 < self.fdr_threshold <= 1:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'total_runs': self.total_runs,
            'inner_group_size': self.inner_group_size,
            'support_threshold': self.support_threshold,
            'fdr_threshold': self.fdr_threshold,
            'epsilon': self.epsilon
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
