from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time

@dataclass
class DesignParameters:
    """Design Parameters"""
    initial_prompt: str
    target_length: int
    project_id: Optional[str] = None
    custom_requirements: Optional[str] = None
    
@dataclass
class StageResult:
    """Stage Result"""
    stage: int
    stage_name: str
    success: bool
    sequence: str
    quality_score: float
    iteration: int
    timestamp: float
    notes: str = ""
    analysis: Optional[Dict[str, Any]] = None
    
@dataclass
class Project:
    """Project Data Model"""
    project_id: str
    parameters: DesignParameters
    stage_results: List[StageResult] = field(default_factory=list)
    final_sequence: Optional[str] = None
    status: str = "in_progress"  # in_progress, completed, failed
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    notes: str = ""
    agent_optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_best_result(self) -> Optional[StageResult]:
        """Get best result"""
        if not self.stage_results:
            return None
        return max(self.stage_results, key=lambda x: x.quality_score)
    
    def get_final_quality_score(self) -> float:
        """Get final quality score"""
        best_result = self.get_best_result()
        return best_result.quality_score if best_result else 0.0
    
    def add_stage_result(self, result: StageResult):
        """Add stage result"""
        self.stage_results.append(result)
        if result.success and result.sequence:
            self.final_sequence = result.sequence