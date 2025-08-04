"""Iteration Optimization Agent - Automatic iterative optimization and parameter adjustment"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import json
from ..models.glm_client import GLMClient
from ..models.evo2_client import Evo2Client
from .quality_agent import QualityAssessmentAgent, QualityAssessment
from ..optimization.llm_optimizer import LLMSequenceOptimizer
from ..utils.logger import get_logger


@dataclass
class IterationParameters:
    """Iteration parameters"""
    max_iterations: int = 5
    quality_threshold: float = 75.0
    improvement_threshold: float = 5.0  # Minimum improvement threshold
    temperature_range: Tuple[float, float] = (0.3, 0.9)
    length_tolerance: int = 10  # Length tolerance
    convergence_patience: int = 2  # Convergence patience
    enable_llm_optimization: bool = True  # Enable LLM optimization


@dataclass
class IterationResult:
    """Single iteration result"""
    iteration: int
    sequence: str
    quality_assessment: QualityAssessment
    parameters_used: Dict[str, Any]
    improvement: float
    strategy: str
    execution_time: float
    success: bool
    notes: str = ""


@dataclass
class OptimizationHistory:
    """Optimization history record"""
    iterations: List[IterationResult]
    best_result: Optional[IterationResult]
    convergence_achieved: bool
    total_time: float
    final_recommendation: str


class IterationOptimizationAgent:
    """Iteration Optimization Agent - Automatically execute sequence optimization iterations"""
    
    def __init__(self, evo2_client: Evo2Client, glm_client: GLMClient, 
                 quality_agent: QualityAssessmentAgent):
        self.evo2_client = evo2_client
        self.glm_client = glm_client
        self.quality_agent = quality_agent
        self.llm_optimizer = LLMSequenceOptimizer(glm_client)
        self.learning_memory = []  # Learning memory
        
    def optimize_sequence(self, initial_sequence: str, target_specs: Dict[str, Any],
                         iteration_params: IterationParameters = None) -> OptimizationHistory:
        """Execute sequence optimization iterations
        
        Args:
            initial_sequence: Initial sequence
            target_specs: Target specifications
            iteration_params: Iteration parameters
            
        Returns:
            Optimization history record
        """
        if iteration_params is None:
            iteration_params = IterationParameters()
            
        start_time = time.time()
        iterations = []
        best_result = None
        current_sequence = initial_sequence
        no_improvement_count = 0
        
        # Get logger
        logger = get_logger()
        
        # Log optimization start
        logger.log_design_start({
            'initial_sequence_length': len(initial_sequence),
            'target_specs': target_specs,
            'iteration_params': asdict(iteration_params)
        })
        
        print(f"🚀 Starting automatic iterative optimization (max iterations: {iteration_params.max_iterations})")
        print(f"📋 Target specs: max length={target_specs.get('max_length', 'N/A')}bp, application={target_specs.get('application', 'N/A')}")
        print(f"⚙️  Iteration params: quality threshold={iteration_params.quality_threshold}, improvement threshold={iteration_params.improvement_threshold}")
        
        for i in range(iteration_params.max_iterations):
            iteration_start = time.time()
            print(f"\n{'='*60}")
            print(f"📊 Iteration {i+1} started ({i+1}/{iteration_params.max_iterations})")
            print(f"{'='*60}")
            
            # 1. Quality assessment
            print(f"🔍 Evaluating sequence quality...")
            quality_assessment = self.quality_agent.assess_sequence_quality(
                current_sequence, target_specs
            )
            
            # Display detailed quality assessment results
            metrics = quality_assessment.metrics
            print(f"📈 Quality assessment results:")
            print(f"   • Overall score: {metrics.overall_score:.1f}/100")
            print(f"   • Functional score: {metrics.functional_score:.1f}/100")
            print(f"   • Expression efficiency score: {metrics.expression_score:.1f}/100")
            print(f"   • Structural stability score: {metrics.structural_score:.1f}/100")
            if metrics.critical_issues:
                print(f"   ⚠️  Critical issues ({len(metrics.critical_issues)} items):")
                for issue in metrics.critical_issues[:3]:  # Only show first 3
                    print(f"      - {issue}")
            else:
                print(f"   ✅ No critical issues")
            
            # Log quality assessment
            logger.log_quality_assessment(current_sequence, asdict(metrics))
            
            # 2. Determine optimization strategy
            strategy = self._determine_optimization_strategy(
                quality_assessment, iterations, target_specs
            )
            print(f"🎯 Optimization strategy: {strategy}")
            
            # 3. Adjust parameters
            optimized_params = self._adjust_parameters(
                strategy, i, iteration_params, quality_assessment
            )
            print(f"⚙️  Parameter settings: temperature={optimized_params.get('temperature', 'N/A'):.2f}, num_sequences={optimized_params.get('num_sequences', 'N/A')}")
            
            # Log iteration start
            logger.log_iteration_start(i+1, strategy, optimized_params)
            
            # 4. Generate improved sequence
            print(f"🧬 Generating improved sequence...")
            improved_sequence, generation_success = self._generate_improved_sequence(
                current_sequence, strategy, optimized_params, target_specs
            )
            
            if generation_success:
                print(f"✅ Sequence generation successful: {len(current_sequence)}bp -> {len(improved_sequence)}bp")
                
                # 5. LLM optimization enhancement
                if iteration_params.enable_llm_optimization:
                    print(f"🤖 Performing LLM intelligent optimization...")
                    llm_result = self.llm_optimizer.optimize_sequence(
                        improved_sequence, target_specs, 
                        optimization_goals=["Improve expression efficiency", "Optimize codon usage", "Reduce secondary structure"]
                    )
                    
                    if llm_result.optimization_score > 0:
                        improved_sequence = llm_result.optimized_sequence
                        print(f"🎯 LLM optimization completed: optimization score={llm_result.optimization_score:.1f}, confidence={llm_result.confidence:.2f}")
                        print(f"   Improvements: {', '.join(llm_result.improvements[:3])}")
                    else:
                        print(f"⚠️  LLM optimization did not produce improvement, keeping EVO2 generated sequence")
            else:
                print(f"❌ Sequence generation failed, keeping original sequence")
            
            # 5. Calculate improvement magnitude
            improvement = 0.0
            if iterations:
                previous_score = iterations[-1].quality_assessment.metrics.overall_score
                improvement = quality_assessment.metrics.overall_score - previous_score
            
            # 6. Record iteration result
            iteration_duration = time.time() - iteration_start
            iteration_result = IterationResult(
                iteration=i+1,
                sequence=improved_sequence if generation_success else current_sequence,
                quality_assessment=quality_assessment,
                parameters_used=optimized_params,
                improvement=improvement,
                strategy=strategy,
                execution_time=iteration_duration,
                success=generation_success,
                notes=self._generate_iteration_notes(quality_assessment, strategy)
            )
            
            iterations.append(iteration_result)
            
            # Display detailed iteration results
            print(f"📊 Iteration results:")
            print(f"   • Improvement: {improvement:+.2f} points")
            print(f"   • New quality score: {quality_assessment.metrics.overall_score:.1f}/100")
            print(f"   • Iteration time: {iteration_duration:.1f} seconds")
            
            # Log iteration result
            logger.log_iteration_result(i+1, {
                'iteration': i+1,
                'strategy': strategy,
                'improvement': improvement,
                'quality_score': quality_assessment.metrics.overall_score,
                'execution_time': iteration_duration,
                'success': generation_success
            })
            
            # 7. Update best result
            if (best_result is None or 
                quality_assessment.metrics.overall_score > best_result.quality_assessment.metrics.overall_score):
                best_result = iteration_result
                no_improvement_count = 0
                print(f"✅ Found better sequence! Score: {quality_assessment.metrics.overall_score:.1f}")
            else:
                no_improvement_count += 1
                print(f"⚠️  No improvement in this iteration (consecutive {no_improvement_count} times)")
            
            # 8. Check convergence conditions
            convergence_check = self._check_convergence(quality_assessment, iteration_params, no_improvement_count)
            
            if convergence_check:
                print(f"\n🎯 Convergence check:")
                if quality_assessment.metrics.overall_score >= iteration_params.quality_threshold:
                    print(f"   ✅ Reached quality threshold ({quality_assessment.metrics.overall_score:.1f} >= {iteration_params.quality_threshold})")
                if no_improvement_count >= iteration_params.convergence_patience:
                    print(f"   ✅ Consecutive no-improvement count reached threshold ({no_improvement_count} >= {iteration_params.convergence_patience})")
                if not quality_assessment.metrics.critical_issues and quality_assessment.metrics.overall_score > 70:
                    print(f"   ✅ No critical issues and high score ({quality_assessment.metrics.overall_score:.1f} > 70)")
                print(f"🎯 Convergence conditions met, ending iterations early")
                break
                
            # 9. Update current sequence
            if generation_success and improvement > 0:
                current_sequence = improved_sequence
                print(f"   ✅ Sequence updated (improvement: {improvement:.2f} points)")
            else:
                print(f"   ❌ Sequence not updated (consecutive no-improvement count: {no_improvement_count})")
                
            print(f"📈 Current best quality score: {best_result.quality_assessment.metrics.overall_score:.1f}/100" if best_result else "📈 No best result yet")
            print(f"⏱️  Cumulative optimization time: {time.time() - start_time:.1f} seconds")
            
            # 10. Learning and memory update
            self._update_learning_memory(iteration_result)
        
        # Generate final recommendation
        final_recommendation = self._generate_final_recommendation(
            iterations, best_result, target_specs
        )
        
        total_time = time.time() - start_time
        convergence_achieved = (best_result and 
                              best_result.quality_assessment.metrics.overall_score >= iteration_params.quality_threshold)
        
        history = OptimizationHistory(
            iterations=iterations,
            best_result=best_result,
            convergence_achieved=convergence_achieved,
            total_time=total_time,
            final_recommendation=final_recommendation
        )
        
        print(f"\n🏁 Optimization completed! Total time: {total_time:.1f} seconds")
        print(f"📈 Best score: {best_result.quality_assessment.metrics.overall_score:.1f}" if best_result else "❌ No valid result found")
        
        return history
    
    def _determine_optimization_strategy(self, quality_assessment: QualityAssessment, 
                                       iterations: List[IterationResult],
                                       target_specs: Dict[str, Any]) -> str:
        """Determine optimization strategy"""
        metrics = quality_assessment.metrics
        
        # Determine strategy based on critical issues
        if metrics.critical_issues:
            critical_keywords = {
                'functional_repair': ['missing promoter', 'missing RBS', 'missing start codon'],
                'length_optimization': ['length', 'exceeds'],
                'composition_adjustment': ['GC content', 'not in'],
                'structure_optimization': ['secondary structure', 'hairpin', 'repeat']
            }
            
            for strategy, keywords in critical_keywords.items():
                for issue in metrics.critical_issues:
                    if any(keyword in issue for keyword in keywords):
                        return strategy
        
        # Determine strategy based on scores
        if metrics.functional_score < 60:
            return 'functional_enhancement'
        elif metrics.expression_score < 60:
            return 'expression_optimization'
        elif metrics.structural_score < 60:
            return 'structure_refinement'
        elif metrics.overall_score < 75:
            return 'general_improvement'
        else:
            return 'fine_tuning'
    
    def _adjust_parameters(self, strategy: str, iteration: int, 
                          iteration_params: IterationParameters,
                          quality_assessment: QualityAssessment) -> Dict[str, Any]:
        """Adjust parameters based on strategy"""
        base_params = {
            'temperature': 0.7,
            'max_length': 140,
            'num_sequences': 3,
            'focus_region': None
        }
        
        # Strategy-specific adjustments
        strategy_adjustments = {
            'functional_repair': {
                'temperature': 0.3,  # Low temperature, conservative generation
                'num_sequences': 5,
                'focus_region': 'functional_elements'
            },
            'length_optimization': {
                'temperature': 0.4,
                'max_length': min(140, len(quality_assessment.sequence) - 10),
                'focus_region': 'non_essential'
            },
            'composition_adjustment': {
                'temperature': 0.5,
                'num_sequences': 4,
                'focus_region': 'gc_content'
            },
            'expression_optimization': {
                'temperature': 0.6,
                'num_sequences': 4,
                'focus_region': 'regulatory_elements'
            },
            'fine_tuning': {
                'temperature': 0.8,  # High temperature, exploratory
                'num_sequences': 2
            }
        }
        
        if strategy in strategy_adjustments:
            base_params.update(strategy_adjustments[strategy])
        
        # Iteration count adjustment
        if iteration > 2:
            base_params['temperature'] = min(0.9, base_params['temperature'] + 0.1)
        
        # Adjustment based on learning memory
        if self.learning_memory:
            successful_params = [mem['params'] for mem in self.learning_memory if mem['success']]
            if successful_params:
                avg_temp = sum(p.get('temperature', 0.7) for p in successful_params) / len(successful_params)
                base_params['temperature'] = (base_params['temperature'] + avg_temp) / 2
        
        return base_params
    
    def _generate_improved_sequence(self, current_sequence: str, strategy: str,
                                  params: Dict[str, Any], target_specs: Dict[str, Any]) -> Tuple[str, bool]:
        """Generate improved sequence"""
        try:
            # Build generation prompt
            prompt = self._build_generation_prompt(current_sequence, strategy, target_specs)
            
            # Call Evo2 to generate sequence
            result = self.evo2_client.generate_sequence(
                prompt=prompt,
                max_tokens=params.get('max_length', 140),  # Fix: use max_tokens instead of max_length
                temperature=params.get('temperature', 0.7)
            )
            
            # Check if result is None to prevent 'NoneType' object is not subscriptable error
            if result and result.get('success') and result.get('sequences'):
                # Select best sequence
                best_sequence = self._select_best_sequence(
                    result['sequences'], current_sequence, target_specs
                )
                return best_sequence, True
            else:
                return current_sequence, False
                
        except Exception as e:
            print(f"❌ Sequence generation failed: {str(e)}")
            return current_sequence, False
    
    def _build_generation_prompt(self, current_sequence: str, strategy: str, 
                               target_specs: Dict[str, Any]) -> str:
        """Build generation prompt"""
        base_prompt = f"Optimize based on the following sequence: {current_sequence}\n\n"
        
        strategy_prompts = {
            'functional_repair': "Repair functional element defects, ensure inclusion of T7 promoter, RBS and ATG start codon",
            'length_optimization': "Optimize sequence length, reduce non-essential elements while maintaining function",
            'composition_adjustment': "Adjust GC content to 40-60% range, use synonymous codon substitution",
            'expression_optimization': "Optimize expression efficiency, improve RBS strength and 5'UTR structure",
            'structure_refinement': "Eliminate harmful secondary structures, avoid hairpins and repeat sequences",
            'fine_tuning': "Perform fine-tuning to improve overall performance"
        }
        
        strategy_instruction = strategy_prompts.get(strategy, "Perform general optimization")
        
        target_info = ""
        if target_specs:
            target_info = f"""
Target requirements:
- Maximum length: {target_specs.get('max_length', 140)} bp
- Application scenario: {target_specs.get('application', 'cell-free expression')}
- Special requirements: {target_specs.get('special_requirements', 'None')}
"""
        
        return f"{base_prompt}{strategy_instruction}\n{target_info}\nGenerate improved DNA sequence:"
    
    def _select_best_sequence(self, sequences: List[str], current_sequence: str,
                            target_specs: Dict[str, Any]) -> str:
        """Select best sequence from candidate sequences"""
        if not sequences:
            return current_sequence
        
        # Quick quality assessment
        sequence_scores = []
        for seq in sequences:
            try:
                assessment = self.quality_agent.assess_sequence_quality(seq, target_specs)
                sequence_scores.append((seq, assessment.metrics.overall_score))
            except:
                sequence_scores.append((seq, 0.0))
        
        # Select sequence with highest score
        if sequence_scores:
            best_seq, best_score = max(sequence_scores, key=lambda x: x[1])
            return best_seq
        
        return sequences[0]  # If evaluation fails, return the first one
    
    def _check_convergence(self, quality_assessment: QualityAssessment,
                          iteration_params: IterationParameters,
                          no_improvement_count: int) -> bool:
        """Check convergence conditions"""
        # Reached quality threshold
        if quality_assessment.metrics.overall_score >= iteration_params.quality_threshold:
            return True
        
        # Consecutive no improvement
        if no_improvement_count >= iteration_params.convergence_patience:
            return True
        
        # No critical issues and high score
        if (not quality_assessment.metrics.critical_issues and 
            quality_assessment.metrics.overall_score > 70):
            return True
        
        return False
    
    def _update_learning_memory(self, iteration_result: IterationResult):
        """Update learning memory"""
        memory_entry = {
            'strategy': iteration_result.strategy,
            'params': iteration_result.parameters_used,
            'success': iteration_result.success and iteration_result.improvement > 0,
            'score': iteration_result.quality_assessment.metrics.overall_score,
            'improvement': iteration_result.improvement
        }
        
        self.learning_memory.append(memory_entry)
        
        # Maintain memory size limit
        if len(self.learning_memory) > 20:
            self.learning_memory = self.learning_memory[-20:]
    
    def _generate_iteration_notes(self, quality_assessment: QualityAssessment, strategy: str) -> str:
        """Generate iteration notes"""
        notes = f"Strategy: {strategy}\n"
        notes += f"Score: {quality_assessment.metrics.overall_score:.1f}\n"
        
        if quality_assessment.metrics.critical_issues:
            notes += f"Critical issues: {len(quality_assessment.metrics.critical_issues)} items\n"
        
        if quality_assessment.needs_iteration:
            notes += f"Recommend continue iteration: {quality_assessment.iteration_strategy}\n"
        
        return notes
    
    def _generate_final_recommendation(self, iterations: List[IterationResult],
                                     best_result: Optional[IterationResult],
                                     target_specs: Dict[str, Any]) -> str:
        """Generate final recommendation"""
        if not best_result:
            return "Optimization failed, recommend checking initial sequence and parameter settings"
        
        recommendation = f"""
🎯 Optimization Results Summary:

Best sequence score: {best_result.quality_assessment.metrics.overall_score:.1f}/100
- Functionality: {best_result.quality_assessment.metrics.functional_score:.1f}
- Expression efficiency: {best_result.quality_assessment.metrics.expression_score:.1f}
- Structural stability: {best_result.quality_assessment.metrics.structural_score:.1f}

Iteration statistics:
- Total iterations: {len(iterations)}
- Successful iterations: {sum(1 for it in iterations if it.success)}
- Average improvement: {sum(it.improvement for it in iterations if it.improvement > 0) / max(1, sum(1 for it in iterations if it.improvement > 0)):.1f}

"""
        
        if best_result.quality_assessment.metrics.critical_issues:
            recommendation += f"\n⚠️  Critical issues still exist:\n"
            for issue in best_result.quality_assessment.metrics.critical_issues[:3]:
                recommendation += f"- {issue}\n"
        
        if best_result.quality_assessment.actionable_recommendations:
            recommendation += f"\n💡 Next step recommendations:\n"
            for rec in best_result.quality_assessment.actionable_recommendations[:3]:
                recommendation += f"- {rec}\n"
        
        return recommendation
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get optimization insights"""
        if not self.learning_memory:
            return {"message": "No learning data available yet"}
        
        successful_strategies = [mem['strategy'] for mem in self.learning_memory if mem['success']]
        strategy_success_rate = {}
        
        for strategy in set(mem['strategy'] for mem in self.learning_memory):
            total = sum(1 for mem in self.learning_memory if mem['strategy'] == strategy)
            success = sum(1 for mem in self.learning_memory if mem['strategy'] == strategy and mem['success'])
            strategy_success_rate[strategy] = success / total if total > 0 else 0
        
        return {
            'total_iterations': len(self.learning_memory),
            'success_rate': sum(1 for mem in self.learning_memory if mem['success']) / len(self.learning_memory) * 100,
            'best_strategy': max(set(successful_strategies), key=successful_strategies.count) if successful_strategies else "None",
            'average_improvement': sum(mem['improvement'] for mem in self.learning_memory if mem['improvement'] > 0) / max(1, sum(1 for mem in self.learning_memory if mem['improvement'] > 0))
        }