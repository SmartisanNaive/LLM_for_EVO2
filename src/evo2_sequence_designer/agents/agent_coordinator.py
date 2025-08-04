"""Agent Coordinator - Unified management and coordination of various intelligent agents"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import time
import json
from ..models.evo2_client import Evo2Client
from ..models.glm_client import GLMClient
from ..analysis.sequence_analyzer import SequenceAnalyzer
from .quality_agent import QualityAssessmentAgent, QualityAssessment
from .iteration_agent import IterationOptimizationAgent, IterationParameters, OptimizationHistory


@dataclass
class AgentConfig:
    """Agent Configuration"""
    enable_quality_assessment: bool = True
    enable_auto_iteration: bool = True
    quality_threshold: float = 75.0
    max_iterations: int = 5
    improvement_threshold: float = 5.0
    real_time_feedback: bool = True
    save_history: bool = True


@dataclass
class CoordinationResult:
    """Coordination Result"""
    success: bool
    final_sequence: str
    quality_assessment: Optional[QualityAssessment]
    optimization_history: Optional[OptimizationHistory]
    execution_time: float
    agent_decisions: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float


class AgentCoordinator:
    """Agent Coordinator - Intelligent coordination of quality assessment and iterative optimization"""
    
    def __init__(self, evo2_client: Evo2Client, glm_client: GLMClient, 
                 sequence_analyzer: SequenceAnalyzer, config: AgentConfig = None):
        self.evo2_client = evo2_client
        self.glm_client = glm_client
        self.sequence_analyzer = sequence_analyzer
        self.config = config or AgentConfig()
        
        # Initialize agents
        self.quality_agent = QualityAssessmentAgent(glm_client, sequence_analyzer)
        self.iteration_agent = IterationOptimizationAgent(evo2_client, glm_client, self.quality_agent)
        
        # Callback functions
        self.progress_callback: Optional[Callable] = None
        self.decision_callback: Optional[Callable] = None
        
        # Execution history
        self.execution_history = []
    
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set progress callback function"""
        self.progress_callback = callback
    
    def set_decision_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set decision callback function"""
        self.decision_callback = callback
    
    def coordinate_optimization(self, initial_sequence: str, target_specs: Dict[str, Any],
                              custom_params: Dict[str, Any] = None) -> CoordinationResult:
        """Coordinate the complete sequence optimization workflow
        
        Args:
            initial_sequence: Initial sequence
            target_specs: Target specifications
            custom_params: Custom parameters
            
        Returns:
            Coordination result
        """
        start_time = time.time()
        agent_decisions = []
        
        self._report_progress("Starting intelligent sequence optimization", 0.0)
        
        try:
            # 1. Initial quality assessment
            self._report_progress("Performing initial quality assessment", 0.1)
            initial_assessment = self.quality_agent.assess_sequence_quality(
                initial_sequence, target_specs
            )
            
            decision = {
                "stage": "initial_assessment",
                "sequence_length": len(initial_sequence),
                "quality_score": initial_assessment.metrics.overall_score,
                "critical_issues": len(initial_assessment.metrics.critical_issues),
                "needs_iteration": initial_assessment.needs_iteration,
                "confidence": initial_assessment.confidence_level
            }
            agent_decisions.append(decision)
            self._report_decision(decision)
            
            # 2. Decide whether iterative optimization is needed
            if not self.config.enable_auto_iteration:
                self._report_progress("Skipping automatic iterative optimization", 1.0)
                return CoordinationResult(
                    success=True,
                    final_sequence=initial_sequence,
                    quality_assessment=initial_assessment,
                    optimization_history=None,
                    execution_time=time.time() - start_time,
                    agent_decisions=agent_decisions,
                    recommendations=initial_assessment.actionable_recommendations,
                    confidence_score=initial_assessment.confidence_level
                )
            
            # 3. Intelligent optimization strategy decision
            optimization_needed = self._decide_optimization_strategy(
                initial_assessment, target_specs
            )
            
            strategy_decision = {
                "stage": "strategy_decision",
                "optimization_needed": optimization_needed["needed"],
                "strategy": optimization_needed["strategy"],
                "reasoning": optimization_needed["reasoning"],
                "estimated_iterations": optimization_needed["estimated_iterations"]
            }
            agent_decisions.append(strategy_decision)
            self._report_decision(strategy_decision)
            
            if not optimization_needed["needed"]:
                self._report_progress("Sequence quality meets requirements, no iteration needed", 1.0)
                return CoordinationResult(
                    success=True,
                    final_sequence=initial_sequence,
                    quality_assessment=initial_assessment,
                    optimization_history=None,
                    execution_time=time.time() - start_time,
                    agent_decisions=agent_decisions,
                    recommendations=initial_assessment.actionable_recommendations,
                    confidence_score=initial_assessment.confidence_level
                )
            
            # 4. Execute iterative optimization
            self._report_progress("Starting automatic iterative optimization", 0.3)
            
            iteration_params = self._prepare_iteration_parameters(
                optimization_needed, custom_params
            )
            
            optimization_history = self.iteration_agent.optimize_sequence(
                initial_sequence, target_specs, iteration_params
            )
            
            # 5. Final quality verification
            self._report_progress("Performing final quality verification", 0.9)
            
            final_sequence = (optimization_history.best_result.sequence 
                            if optimization_history.best_result 
                            else initial_sequence)
            
            final_assessment = self.quality_agent.assess_sequence_quality(
                final_sequence, target_specs
            )
            
            # 6. Generate comprehensive recommendations
            recommendations = self._generate_comprehensive_recommendations(
                initial_assessment, final_assessment, optimization_history
            )
            
            # 7. Calculate confidence score
            confidence_score = self._calculate_overall_confidence(
                final_assessment, optimization_history
            )
            
            final_decision = {
                "stage": "final_result",
                "initial_score": initial_assessment.metrics.overall_score,
                "final_score": final_assessment.metrics.overall_score,
                "improvement": final_assessment.metrics.overall_score - initial_assessment.metrics.overall_score,
                "iterations_used": len(optimization_history.iterations),
                "convergence_achieved": optimization_history.convergence_achieved,
                "confidence": confidence_score
            }
            agent_decisions.append(final_decision)
            self._report_decision(final_decision)
            
            self._report_progress("Optimization completed", 1.0)
            
            result = CoordinationResult(
                success=True,
                final_sequence=final_sequence,
                quality_assessment=final_assessment,
                optimization_history=optimization_history,
                execution_time=time.time() - start_time,
                agent_decisions=agent_decisions,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
            # Save execution history
            if self.config.save_history:
                self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # Detailed error analysis
            error_type = type(e).__name__
            error_message = str(e)
            
            # Provide specific diagnostic information based on error type
            diagnostic_info = self._analyze_error(e, error_type, error_message)
            
            error_decision = {
                "stage": "error",
                "error_type": error_type,
                "error_message": error_message,
                "diagnostic_info": diagnostic_info,
                "fallback_sequence": initial_sequence,
                "error_details": error_details,
                "timestamp": time.time()
            }
            agent_decisions.append(error_decision)
            
            # Log detailed error information
            self._log_coordination_error(error_decision)
            
            return CoordinationResult(
                success=False,
                final_sequence=initial_sequence,
                quality_assessment=None,
                optimization_history=None,
                execution_time=time.time() - start_time,
                agent_decisions=agent_decisions,
                recommendations=[
                    f"Agent coordination optimization failed: {error_message}",
                    f"Error type: {error_type}",
                    *diagnostic_info.get('recommendations', [])
                ],
                confidence_score=0.0,
                error_message=error_message
            )
    
    def _decide_optimization_strategy(self, assessment: QualityAssessment, 
                                    target_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent optimization strategy decision"""
        metrics = assessment.metrics
        
        # Basic decision logic
        needs_optimization = (
            metrics.overall_score < self.config.quality_threshold or
            len(metrics.critical_issues) > 0 or
            assessment.needs_iteration
        )
        
        if not needs_optimization:
            return {
                "needed": False,
                "strategy": "none",
                "reasoning": "Sequence quality meets standards, no optimization needed",
                "estimated_iterations": 0
            }
        
        # Determine optimization strategy
        if len(metrics.critical_issues) > 2:
            strategy = "aggressive_repair"
            estimated_iterations = min(self.config.max_iterations, 4)
            reasoning = "Multiple critical issues exist, aggressive repair needed"
        elif metrics.functional_score < 60:
            strategy = "functional_focus"
            estimated_iterations = 3
            reasoning = "Low functional score, focus on functional optimization"
        elif metrics.expression_score < 60:
            strategy = "expression_focus"
            estimated_iterations = 3
            reasoning = "Low expression efficiency, focus on expression optimization"
        elif metrics.overall_score < 70:
            strategy = "comprehensive_improvement"
            estimated_iterations = 4
            reasoning = "Overall quality needs comprehensive improvement"
        else:
            strategy = "fine_tuning"
            estimated_iterations = 2
            reasoning = "Good quality, perform fine-tuning"
        
        return {
            "needed": True,
            "strategy": strategy,
            "reasoning": reasoning,
            "estimated_iterations": estimated_iterations
        }
    
    def _prepare_iteration_parameters(self, optimization_strategy: Dict[str, Any],
                                    custom_params: Dict[str, Any] = None) -> IterationParameters:
        """Prepare iteration parameters"""
        base_params = IterationParameters(
            max_iterations=optimization_strategy["estimated_iterations"],
            quality_threshold=self.config.quality_threshold,
            improvement_threshold=self.config.improvement_threshold
        )
        
        # Adjust parameters based on strategy
        strategy = optimization_strategy["strategy"]
        if strategy == "aggressive_repair":
            base_params.convergence_patience = 1
            base_params.temperature_range = (0.2, 0.6)
        elif strategy == "fine_tuning":
            base_params.temperature_range = (0.6, 0.9)
            base_params.improvement_threshold = 2.0
        
        # Apply custom parameters
        if custom_params:
            for key, value in custom_params.items():
                if hasattr(base_params, key):
                    setattr(base_params, key, value)
        
        return base_params
    
    def _analyze_error(self, exception: Exception, error_type: str, error_message: str) -> Dict[str, Any]:
        """Analyze errors and provide diagnostic information"""
        diagnostic_info = {
            "category": "unknown",
            "severity": "medium",
            "possible_causes": [],
            "recommendations": [],
            "recovery_actions": []
        }
        
        # API related errors
        if "API" in error_message or "key" in error_message.lower() or "401" in error_message:
            diagnostic_info.update({
                "category": "api_authentication",
                "severity": "high",
                "possible_causes": [
                    "Invalid or expired API key",
                "Incorrect API key configuration",
                "API service temporarily unavailable"
                ],
                "recommendations": [
                    "Check API key configuration: python main.py api-config",
                "Test API connection: python main.py test",
                "Verify API key validity"
                ],
                "recovery_actions": [
                    "Reconfigure API key",
                "Contact API service provider",
                "Use backup API endpoint"
                ]
            })
        
        # Network connection errors
        elif any(keyword in error_message.lower() for keyword in ["timeout", "connection", "network", "unreachable"]):
            diagnostic_info.update({
                "category": "network_connectivity",
                "severity": "high",
                "possible_causes": [
                    "Unstable network connection",
                "Firewall blocking connection",
                "DNS resolution issues",
                "Proxy configuration problems"
                ],
                "recommendations": [
                    "Check network connection status",
                "Try using VPN or proxy",
                "Check firewall settings",
                "Retry later"
                ],
                "recovery_actions": [
                    "Restart network connection",
                "Switch network environment",
                "Configure proxy server"
                ]
            })
        
        # Memory or resource errors
        elif any(keyword in error_message.lower() for keyword in ["memory", "resource", "limit"]):
            diagnostic_info.update({
                "category": "resource_limitation",
                "severity": "medium",
                "possible_causes": [
                    "Insufficient system memory",
                "API call rate limit",
                "Sequence length exceeds limit"
                ],
                "recommendations": [
                    "Reduce concurrent request count",
                "Lower sequence complexity",
                "Increase system memory",
                "Process requests in batches"
                ],
                "recovery_actions": [
                    "Restart program to free memory",
                "Adjust optimization parameters",
                "Use simplified mode"
                ]
            })
        
        # Data format errors
        elif any(keyword in error_type.lower() for keyword in ["json", "parse", "format", "decode"]):
            diagnostic_info.update({
                "category": "data_format",
                "severity": "medium",
                "possible_causes": [
                    "Abnormal API response format",
                "Data parsing error",
                "Encoding issues"
                ],
                "recommendations": [
                    "Check API response format",
                "Verify input data format",
                "Retry request"
                ],
                "recovery_actions": [
                    "Retry with default parameters",
                "Check data encoding",
                "Contact technical support"
                ]
            })
        
        # Permission errors
        elif "permission" in error_message.lower() or "403" in error_message:
            diagnostic_info.update({
                "category": "permission_denied",
                "severity": "high",
                "possible_causes": [
                    "Insufficient API permissions",
                "Account quota exhausted",
                "Service access restricted"
                ],
                "recommendations": [
                    "Check API account status",
                "Verify service permissions",
                "Contact service provider"
                ],
                "recovery_actions": [
                    "Upgrade API service plan",
                "Wait for quota reset",
                "Use backup account"
                ]
            })
        
        return diagnostic_info
    
    def _log_coordination_error(self, error_decision: Dict[str, Any]):
        """Log detailed coordination error information"""
        try:
            import logging
            import os
            
            # Create log directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Configure error logger
            error_logger = logging.getLogger("agent_coordinator_errors")
            if not error_logger.handlers:
                handler = logging.FileHandler(log_dir / "agent_coordination_errors.log")
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                error_logger.addHandler(handler)
                error_logger.setLevel(logging.ERROR)
            
            # Log error information
            error_logger.error(
                f"Agent coordination failed - Type: {error_decision['error_type']} | "
                f"Message: {error_decision['error_message']} | "
                f"Diagnosis: {error_decision['diagnostic_info']['category']} | "
                f"Time: {error_decision['timestamp']}"
            )
            
            # Log detailed error stack
            if 'error_details' in error_decision:
                error_logger.error(f"Detailed error stack:\n{error_decision['error_details']}")
                
        except Exception as log_error:
            # If logging fails, at least print to console
            print(f"⚠️ Unable to log error: {log_error}")
            print(f"Original error: {error_decision['error_message']}")
    
    def _generate_comprehensive_recommendations(self, initial_assessment: QualityAssessment,
                                              final_assessment: QualityAssessment,
                                              optimization_history: OptimizationHistory) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Recommendations based on improvement level
        improvement = final_assessment.metrics.overall_score - initial_assessment.metrics.overall_score
        
        if improvement > 10:
            recommendations.append("✅ Significant optimization effect, sequence quality greatly improved")
        elif improvement > 5:
            recommendations.append("✅ Good optimization effect, sequence quality improved")
        elif improvement > 0:
            recommendations.append("⚠️ Limited optimization effect, suggest adjusting strategy")
        else:
            recommendations.append("❌ No optimization effect, suggest re-evaluating initial sequence")
        
        # Recommendations based on final quality
        if final_assessment.metrics.overall_score >= 85:
            recommendations.append("🎯 Excellent sequence quality, ready for direct experimental use")
        elif final_assessment.metrics.overall_score >= 75:
            recommendations.append("✅ Good sequence quality, recommend small-scale validation")
        elif final_assessment.metrics.overall_score >= 60:
            recommendations.append("⚠️ Average sequence quality, recommend further optimization")
        else:
            recommendations.append("❌ Poor sequence quality, not recommended for direct use")
        
        # Add specific technical recommendations
        recommendations.extend(final_assessment.actionable_recommendations[:3])
        
        # Recommendations based on optimization history
        if optimization_history and optimization_history.iterations:
            successful_iterations = sum(1 for it in optimization_history.iterations if it.success)
            if successful_iterations < len(optimization_history.iterations) / 2:
                recommendations.append("💡 Suggest checking Evo2 model parameters or trying different optimization strategies")
        
        return recommendations
    
    def _calculate_overall_confidence(self, final_assessment: QualityAssessment,
                                    optimization_history: OptimizationHistory) -> float:
        """Calculate overall confidence score"""
        base_confidence = final_assessment.confidence_level
        
        # Adjust based on quality score
        score_factor = final_assessment.metrics.overall_score / 100.0
        
        # Adjust based on optimization history
        history_factor = 1.0
        if optimization_history and optimization_history.iterations:
            successful_rate = sum(1 for it in optimization_history.iterations if it.success) / len(optimization_history.iterations)
            history_factor = 0.7 + 0.3 * successful_rate
        
        # Adjust based on critical issues
        issue_factor = max(0.5, 1.0 - len(final_assessment.metrics.critical_issues) * 0.1)
        
        overall_confidence = base_confidence * score_factor * history_factor * issue_factor
        return min(1.0, max(0.0, overall_confidence))
    
    def _report_progress(self, message: str, progress: float):
        """Report progress"""
        if self.config.real_time_feedback and self.progress_callback:
            self.progress_callback(message, progress)
        else:
            print(f"[{progress*100:.0f}%] {message}")
    
    def _report_decision(self, decision: Dict[str, Any]):
        """Report decision"""
        if self.config.real_time_feedback and self.decision_callback:
            self.decision_callback(decision)
        else:
            print(f"🤖 Agent decision: {decision.get('stage', 'unknown')} - {decision}")
    
    def batch_optimize_sequences(self, sequences: List[str], target_specs: Dict[str, Any],
                               custom_params: Dict[str, Any] = None) -> List[CoordinationResult]:
        """Batch optimize sequences"""
        results = []
        
        for i, sequence in enumerate(sequences):
            self._report_progress(f"Optimizing sequence {i+1}/{len(sequences)}", i / len(sequences))
            
            result = self.coordinate_optimization(sequence, target_specs, custom_params)
            results.append(result)
        
        self._report_progress("Batch optimization completed", 1.0)
        return results
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        if not self.execution_history:
            return {"message": "No execution history"}
        
        successful_runs = [r for r in self.execution_history if r.success]
        
        return {
            "total_runs": len(self.execution_history),
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / len(self.execution_history),
            "average_improvement": sum(
                r.agent_decisions[-1].get("improvement", 0) 
                for r in successful_runs
            ) / max(1, len(successful_runs)),
            "average_execution_time": sum(r.execution_time for r in self.execution_history) / len(self.execution_history),
            "average_confidence": sum(r.confidence_score for r in successful_runs) / max(1, len(successful_runs))
        }
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get optimization insights
        
        Returns:
            Optimization insight information, including strategy success rates, average improvement levels, and other statistics
        """
        return self.iteration_agent.get_optimization_insights()
    
    def export_optimization_report(self, result: CoordinationResult) -> str:
        """Export optimization report"""
        report = f"""
# Sequence Optimization Report

## Basic Information
- Execution time: {result.execution_time:.2f} seconds
- Final sequence length: {len(result.final_sequence)}bp
- Confidence score: {result.confidence_score:.2f}

## Quality Assessment
"""
        
        if result.quality_assessment:
            qa = result.quality_assessment
            report += f"""
- Overall score: {qa.metrics.overall_score:.1f}/100
- Functional score: {qa.metrics.functional_score:.1f}/100
- Expression efficiency score: {qa.metrics.expression_score:.1f}/100
- Structural stability score: {qa.metrics.structural_score:.1f}/100
- Number of critical issues: {len(qa.metrics.critical_issues)}
"""
        
        if result.optimization_history:
            oh = result.optimization_history
            report += f"""

## Optimization History
- Number of iterations: {len(oh.iterations)}
- Convergence status: {'Converged' if oh.convergence_achieved else 'Not converged'}
- Best iteration: #{oh.best_result.iteration}
"""
        
        report += f"""

## Recommendations
"""
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""

## Final Sequence
```
{result.final_sequence}
```
"""
        
        return report