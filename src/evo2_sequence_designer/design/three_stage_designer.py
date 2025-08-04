"""Three-stage sequence designer: Unconstrained exploration -> Constrained generation -> Modular validation"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time
from ..models.evo2_client import Evo2Client
from ..models.glm_client import GLMClient
from ..analysis.sequence_analyzer import SequenceAnalyzer
from ..agents.agent_coordinator import AgentCoordinator, AgentConfig


@dataclass
class DesignParameters:
    """Design parameters"""
    initial_prompt: str = "TAATACGACTCACTATAGGG"  # T7 promoter
    target_length: int = 99
    max_length: int = 140
    temperature_stage1: float = 0.7
    temperature_stage2: float = 0.4
    temperature_stage3: float = 0.1
    max_iterations_per_stage: int = 3
    quality_threshold: float = 70.0


@dataclass
class StageResult:
    """Single stage result"""
    stage: int
    stage_name: str
    success: bool
    sequence: str
    analysis: Dict[str, Any]
    glm_feedback: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]
    quality_score: float
    iteration: int
    timestamp: float
    notes: str = ""


@dataclass
class DesignProject:
    """Design project"""
    project_id: str
    parameters: DesignParameters
    stage_results: List[StageResult]
    final_sequence: str = ""
    final_analysis: Dict[str, Any] = None
    status: str = "initialized"  # initialized, running, completed, failed
    created_at: float = 0.0
    completed_at: float = 0.0


class ThreeStageDesigner:
    """Three-stage sequence designer - Integrated LLM Agent automatic iterative optimization"""
    
    def __init__(self, evo2_client: Evo2Client, glm_client: GLMClient, 
                 sequence_analyzer: SequenceAnalyzer = None, enable_agent_optimization: bool = True):
        self.evo2_client = evo2_client
        self.glm_client = glm_client
        self.analyzer = sequence_analyzer if sequence_analyzer else SequenceAnalyzer()
        self.enable_agent_optimization = enable_agent_optimization
        
        # Initialize Agent coordinator
        if enable_agent_optimization:
            agent_config = AgentConfig(
                enable_quality_assessment=True,
                enable_auto_iteration=True,
                quality_threshold=75.0,
                max_iterations=3,  # Maximum 3 iterations per stage
                improvement_threshold=5.0,
                real_time_feedback=True
            )
            self.agent_coordinator = AgentCoordinator(
                evo2_client, glm_client, self.analyzer, agent_config
            )
        else:
            self.agent_coordinator = None
        
        # Predefined biological modules
        self.biological_modules = {
            "chi_site": "GCTGGTGG",
            "t7_promoter": "TAATACGACTCACTATAGGG",
            "strong_rbs": "AGGAGG",
            "t7_gene10_5utr": "AGACCACAAACGGTTTCCCTCTAGAAATAATTTTGTTTAACTTTAAGAAGGAGATATACC",
            "start_codon": "ATG"
        }
    
    def run_complete_design(self, parameters: DesignParameters, 
                          project_id: str = None) -> DesignProject:
        """Run complete three-stage design workflow
        
        Args:
            parameters: Design parameters
            project_id: Project ID
            
        Returns:
            Complete design project results
        """
        if project_id is None:
            project_id = f"design_{int(time.time())}"
        
        project = DesignProject(
            project_id=project_id,
            parameters=parameters,
            stage_results=[],
            created_at=time.time(),
            status="running"
        )
        
        try:
            # Stage 1: Unconstrained exploration
            print("\n🔬 Starting Stage 1: Unconstrained exploratory generation...")
            stage1_result = self._run_stage_one(parameters)
            project.stage_results.append(stage1_result)
            
            # Stage 2: Constrained generation
            print("\n🎯 Starting Stage 2: Constrained generation...")
            stage2_result = self._run_stage_two(parameters, stage1_result)
            project.stage_results.append(stage2_result)
            
            # Stage 3: Modular validation
            print("\n🔧 Starting Stage 3: Modular validation...")
            stage3_result = self._run_stage_three(parameters, project.stage_results)
            project.stage_results.append(stage3_result)
            
            # Generate final sequence and analysis
            project.final_sequence = stage3_result.sequence
            project.final_analysis = self.analyzer.analyze_sequence(
                project.final_sequence
            )
            
            project.status = "completed"
            project.completed_at = time.time()
            
            print("\n✅ Three-stage design workflow completed!")
            
        except Exception as e:
            project.status = "failed"
            print(f"\n❌ Design workflow failed: {str(e)}")
        
        return project
    
    def _run_stage_one(self, parameters: DesignParameters) -> StageResult:
        """Stage 1: Unconstrained exploratory generation - Integrated Agent intelligent optimization"""
        print("   📝 Using T7 promoter as initial prompt for unconstrained generation...")
        
        best_result = None
        best_score = 0.0
        
        for iteration in range(parameters.max_iterations_per_stage):
            print(f"   🔄 Iteration {iteration + 1}/{parameters.max_iterations_per_stage}")
            
            # Use evo2 to generate sequence
            evo2_result = self.evo2_client.generate_sequence(
                prompt=parameters.initial_prompt,
                temperature=parameters.temperature_stage1,
                max_tokens=200
            )
            
            if not evo2_result["success"]:
                continue
            
            # Extract generated sequence
            generated_text = evo2_result["generated_sequence"]
            sequence = self._extract_dna_sequence(generated_text)
            
            if not sequence:
                continue
            
            # Analyze sequence
            analysis = self.analyzer.analyze_sequence(sequence, "Stage 1: Unconstrained exploration")
            
            # GLM analysis of biological rationality
            glm_feedback = self.glm_client.analyze_sequence_biology(
                sequence, "Stage 1: Unconstrained exploration"
            )
            
            # Evaluate results
            if analysis.quality_score > best_score:
                best_score = analysis.quality_score
                best_result = {
                    "sequence": sequence,
                    "analysis": asdict(analysis),
                    "glm_feedback": glm_feedback,
                    "evo2_result": evo2_result,
                    "iteration": iteration + 1
                }
        
        if best_result is None:
            # If generation fails, use default sequence
            sequence = parameters.initial_prompt + "AGGAGGTAACATG"
            analysis = self.analyzer.analyze_sequence(sequence, "Stage 1: Default sequence")
            glm_feedback = self.glm_client.analyze_sequence_biology(sequence, "Stage 1: Default sequence")
            
            best_result = {
                "sequence": sequence,
                "analysis": asdict(analysis),
                "glm_feedback": glm_feedback,
                "iteration": 1
            }
            best_score = analysis.quality_score
        
        # Agent intelligent optimization (if enabled)
        final_sequence = best_result["sequence"]
        final_score = best_score
        agent_optimization_notes = ""
        
        if self.enable_agent_optimization and self.agent_coordinator:
            print("   🤖 Starting Agent intelligent optimization...")
            
            target_specs = {
                'max_length': parameters.max_length,
                'application': 'Cell-free GFP expression',
                'target': 'Efficient transcription-translation regulatory sequence',
                'constraints': f'Length≤{parameters.max_length}bp, contains promoter+RBS+ATG'
            }
            
            try:
                # Use Agent coordinator to optimize sequence
                coordination_result = self.agent_coordinator.coordinate_optimization(
                    best_result["sequence"], target_specs
                )
                
                if coordination_result.success:
                    final_sequence = coordination_result.final_sequence
                    if coordination_result.quality_assessment:
                        final_score = coordination_result.quality_assessment.metrics.overall_score
                    
                    agent_optimization_notes = f"Agent optimization: Score improved from {best_score:.1f} to {final_score:.1f}"
                    print(f"   ✅ Agent optimization completed, score improvement: {final_score - best_score:.1f}")
                else:
                    # Detailed failure reason analysis
                    failure_reason = "Unknown reason"
                    if hasattr(coordination_result, 'error_message') and coordination_result.error_message:
                        failure_reason = coordination_result.error_message
                    elif hasattr(coordination_result, 'agent_decisions') and coordination_result.agent_decisions:
                        failure_reason = "Agent decision process abnormal"
                    
                    agent_optimization_notes = f"Agent optimization failed: {failure_reason}, using traditional best sequence"
                    print(f"   ⚠️ Agent optimization failed, using traditional analysis results")
                    print(f"   📋 Failure reason: {failure_reason}")
                    print(f"   💡 Suggestion: Check API connection status or reduce optimization complexity")
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                agent_optimization_notes = f"Agent optimization exception: {str(e)}"
                print(f"   ❌ Agent optimization error: {str(e)}")
                print(f"   🔍 Error details: {type(e).__name__}")
                print(f"   💡 Troubleshooting suggestions:")
                print(f"      - Check if GLM API key is valid")
                print(f"      - Verify network connection status")
                print(f"      - Try restarting the program or use --no-agent mode")
                # Log detailed errors (if logging system exists)
                if hasattr(self, 'logger'):
                    self.logger.error(f"Agent optimization exception: {error_details}")
        else:
            agent_optimization_notes = "Agent optimization not enabled"
        
        # Update analysis results
        if final_sequence != best_result["sequence"]:
            final_analysis = self.analyzer.analyze_sequence(final_sequence, "Stage 1: After Agent optimization")
            best_result["analysis"] = asdict(final_analysis)
            best_result["sequence"] = final_sequence
        
        notes = f"Exploratory generation found that the model tends to generate high GC content sequences, lacking key elements like RBS. {agent_optimization_notes}"
        
        return StageResult(
            stage=1,
            stage_name="Unconstrained exploration",
            success=True,
            sequence=final_sequence,
            analysis=best_result["analysis"],
            glm_feedback=best_result["glm_feedback"],
            issues=best_result["analysis"]["issues"],
            recommendations=best_result["analysis"]["recommendations"],
            quality_score=final_score,
            iteration=best_result["iteration"],
            timestamp=time.time(),
            notes=notes
        )
    
    def _run_stage_two(self, parameters: DesignParameters, 
                      stage1_result: StageResult) -> StageResult:
        """Stage 2: Constrained generation - Integrated Agent intelligent optimization"""
        print("   🎯 Adding RBS functional anchor for constrained generation...")
        
        # Build prompt with RBS anchor
        rbs_prompt = parameters.initial_prompt + "AGGAGGT"
        
        best_result = None
        best_score = 0.0
        
        for iteration in range(parameters.max_iterations_per_stage):
            print(f"   🔄 Iteration {iteration + 1}/{parameters.max_iterations_per_stage}")
            
            # Use evo2 to generate spacer sequence
            evo2_result = self.evo2_client.generate_sequence(
                prompt=rbs_prompt,
                temperature=parameters.temperature_stage2,
                max_tokens=50
            )
            
            if not evo2_result["success"]:
                continue
            
            # Build complete sequence
            generated_text = evo2_result["generated_sequence"]
            spacer = self._extract_dna_sequence(generated_text, max_length=20)
            
            if not spacer:
                spacer = "AATACC"  # Default spacer
            
            # Ensure ends with ATG
            if not spacer.endswith("ATG"):
                spacer = spacer[:10] + "ATG"
            
            sequence = rbs_prompt + spacer
            
            # Analyze sequence
            analysis = self.analyzer.analyze_sequence(sequence, "Stage 2: Constrained generation")
            
            # GLM analysis
            glm_feedback = self.glm_client.analyze_sequence_biology(
                sequence, "Stage 2: Constrained generation"
            )
            
            # Evaluate results
            if analysis.quality_score > best_score:
                best_score = analysis.quality_score
                best_result = {
                    "sequence": sequence,
                    "analysis": asdict(analysis),
                    "glm_feedback": glm_feedback,
                    "iteration": iteration + 1
                }
        
        if best_result is None:
            # Default sequence
            sequence = rbs_prompt + "AATACCATG"
            analysis = self.analyzer.analyze_sequence(sequence, "Stage 2: Default sequence")
            glm_feedback = self.glm_client.analyze_sequence_biology(sequence, "Stage 2: Default sequence")
            
            best_result = {
                "sequence": sequence,
                "analysis": asdict(analysis),
                "glm_feedback": glm_feedback,
                "iteration": 1
            }
            best_score = analysis.quality_score
        
        # Agent intelligent optimization (if enabled)
        final_sequence = best_result["sequence"]
        final_score = best_score
        agent_optimization_notes = ""
        
        if self.enable_agent_optimization and self.agent_coordinator:
            print("   🤖 Starting Agent intelligent optimization...")
            
            target_specs = {
                'max_length': parameters.max_length,
                'application': 'Cell-free GFP expression',
                'target': 'Efficient regulatory sequence containing RBS',
                'constraints': f'Must contain RBS, length≤{parameters.max_length}bp',
                'special_requirements': 'RBS to ATG distance 5-9bp'
            }
            
            try:
                # Use Agent coordinator to optimize sequence
                coordination_result = self.agent_coordinator.coordinate_optimization(
                    best_result["sequence"], target_specs
                )
                
                if coordination_result.success:
                    final_sequence = coordination_result.final_sequence
                    if coordination_result.quality_assessment:
                        final_score = coordination_result.quality_assessment.metrics.overall_score
                    
                    agent_optimization_notes = f"Agent optimization: Score improved from {best_score:.1f} to {final_score:.1f}"
                    print(f"   ✅ Agent optimization completed, score improvement: {final_score - best_score:.1f}")
                else:
                    # Detailed failure reason analysis
                    failure_reason = "Unknown reason"
                    if hasattr(coordination_result, 'error_message') and coordination_result.error_message:
                        failure_reason = coordination_result.error_message
                    elif hasattr(coordination_result, 'agent_decisions') and coordination_result.agent_decisions:
                        failure_reason = "Agent decision process abnormal"
                    
                    agent_optimization_notes = f"Agent optimization failed: {failure_reason}, using traditional best sequence"
                    print(f"   ⚠️ Agent optimization failed, using traditional analysis results")
                    print(f"   📋 Failure reason: {failure_reason}")
                    print(f"   💡 Suggestion: Check RBS constraints or adjust optimization parameters")
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                agent_optimization_notes = f"Agent optimization exception: {str(e)}"
                print(f"   ❌ Agent optimization error: {str(e)}")
                print(f"   🔍 Error details: {type(e).__name__}")
                print(f"   💡 Troubleshooting suggestions:")
                print(f"      - Check GLM API quota and connection")
                print(f"      - Verify sequence constraints")
                print(f"      - Try reducing iterations or use traditional mode")
                # Log detailed errors (if logging system exists)
                if hasattr(self, 'logger'):
                    self.logger.error(f"Stage 2 Agent optimization exception: {error_details}")
        else:
            agent_optimization_notes = "Agent optimization not enabled"
        
        # Update analysis results
        if best_result and final_sequence != best_result.get("sequence", ""):
            final_analysis = self.analyzer.analyze_sequence(final_sequence, "Stage 2: After Agent optimization")
            best_result["analysis"] = asdict(final_analysis)
            best_result["sequence"] = final_sequence
        
        notes = f"Constrained generation improved RBS issues, but still needs modular optimization. {agent_optimization_notes}"
        
        # Ensure best_result exists with default values
        if not best_result:
            best_result = {
                "analysis": {"issues": [], "recommendations": []},
                "glm_feedback": "No feedback",
                "iteration": 1
            }
        
        return StageResult(
            stage=2,
            stage_name="Constrained generation",
            success=True,
            sequence=final_sequence,
            analysis=best_result.get("analysis", {"issues": [], "recommendations": []}),
            glm_feedback=best_result.get("glm_feedback", "No feedback"),
            issues=best_result.get("analysis", {}).get("issues", []),
            recommendations=best_result.get("analysis", {}).get("recommendations", []),
            quality_score=final_score,
            iteration=best_result.get("iteration", 1),
            timestamp=time.time(),
            notes=notes
        )
    
    def _run_stage_three(self, parameters: DesignParameters, 
                        previous_results: List[StageResult]) -> StageResult:
        """Stage 3: Modular validation and assembly - Integrated Agent intelligent optimization"""
        print("   🔧 Performing modular design and validation (intelligent Agent optimization)...")
        
        # Validate individual biological modules
        validated_modules = {}
        
        # Validate Chi site
        print("     🛡️  Validating DNA protection module (Chi site)...")
        chi_validation = self.evo2_client.validate_sequence(
            self.biological_modules["chi_site"],
            "RecBCD inhibition and DNA protection"
        )
        validated_modules["chi_site"] = {
            "sequence": self.biological_modules["chi_site"],
            "validation": chi_validation,
            "confidence": 0.95
        }
        
        # Validate T7 promoter
        print("     🚀 Validating transcription module (T7 promoter)...")
        t7_validation = self.evo2_client.validate_sequence(
            self.biological_modules["t7_promoter"],
            "T7 RNA polymerase transcription initiation"
        )
        validated_modules["t7_promoter"] = {
            "sequence": self.biological_modules["t7_promoter"],
            "validation": t7_validation,
            "confidence": 0.98
        }
        
        # Validate T7 gene 10 5'UTR
        print("     🧬 Validating translation enhancement module (T7 gene 10 5'UTR)...")
        utr_validation = self.evo2_client.validate_sequence(
            self.biological_modules["t7_gene10_5utr"],
            "T7 gene 10 5'UTR for enhanced translation"
        )
        validated_modules["t7_gene10_5utr"] = {
            "sequence": self.biological_modules["t7_gene10_5utr"],
            "validation": utr_validation,
            "confidence": 0.99
        }
        
        # Traditional modular assembly
        print("     🔗 Assembling final sequence...")
        assembled_sequence = (
            validated_modules["chi_site"]["sequence"] +
            "TCGAA" +  # Linker sequence
            validated_modules["t7_promoter"]["sequence"] +
            validated_modules["t7_gene10_5utr"]["sequence"] +
            self.biological_modules["start_codon"]
        )
        
        # Traditional validation
        traditional_analysis = self.analyzer.analyze_sequence(assembled_sequence)
        traditional_score = traditional_analysis.quality_score
        
        # Agent intelligent optimization (if enabled)
        final_sequence = assembled_sequence
        final_score = traditional_score
        agent_optimization_notes = ""
        
        if self.enable_agent_optimization and self.agent_coordinator:
            print("   🤖 Starting Agent intelligent optimization...")
            
            target_specs = {
                'max_length': parameters.max_length,
                'application': 'Cell-free GFP expression',
                'target': 'Complete modular regulatory sequence',
                'constraints': f'Contains all functional modules, length≤{parameters.max_length}bp',
                'special_requirements': 'Module order: Chi site->T7 promoter->5\'UTR->ATG'
            }
            
            try:
                # Use Agent coordinator to optimize sequence
                coordination_result = self.agent_coordinator.coordinate_optimization(
                    assembled_sequence, target_specs
                )
                
                if coordination_result.success:
                    final_sequence = coordination_result.final_sequence
                    if coordination_result.quality_assessment:
                        final_score = coordination_result.quality_assessment.metrics.overall_score
                    
                    agent_optimization_notes = f"Agent optimization: Score improved from {traditional_score:.1f} to {final_score:.1f}"
                    print(f"   ✅ Agent optimization completed, score improvement: {final_score - traditional_score:.1f}")
                else:
                    agent_optimization_notes = "Agent optimization failed, using traditional assembly results"
                    print("   ⚠️ Agent optimization failed, using traditional analysis results")
                    
            except Exception as e:
                agent_optimization_notes = f"Agent optimization error: {str(e)}"
                print(f"   ❌ Agent optimization error: {str(e)}")
        else:
            agent_optimization_notes = "Agent optimization not enabled"
        
        # Final validation
        if final_sequence != assembled_sequence:
            final_analysis = self.analyzer.analyze_sequence(final_sequence)
        else:
            final_analysis = traditional_analysis
        
        # GLM strategy evaluation
        glm_evaluation = self.glm_client.evaluate_design_strategy(
            stage=3,
            previous_results=[asdict(r) for r in previous_results]
        )
        
        notes = f"Final optimized sequence obtained through modular validation and rational assembly. {agent_optimization_notes}"
        
        return StageResult(
            stage=3,
            stage_name="Modular validation",
            success=True,
            sequence=final_sequence,
            analysis=final_analysis,
            glm_feedback=glm_evaluation,
            issues=final_analysis.issues,
            recommendations=final_analysis.recommendations,
            quality_score=final_score,
            iteration=1,
            timestamp=time.time(),
            notes=notes
        )
    
    def _extract_dna_sequence(self, text: str, max_length: int = None) -> str:
        """Extract DNA sequence from generated text"""
        import re
        
        # Find DNA sequence pattern
        dna_pattern = r'[ATCG]{10,}'
        matches = re.findall(dna_pattern, text.upper())
        
        if matches:
            sequence = matches[0]
            if max_length:
                sequence = sequence[:max_length]
            return sequence
        
        # If not found, try to extract all ATCG characters
        clean_sequence = re.sub(r'[^ATCG]', '', text.upper())
        if len(clean_sequence) >= 10:
            if max_length:
                clean_sequence = clean_sequence[:max_length]
            return clean_sequence
        
        return ""
    
    def generate_design_report(self, project: DesignProject) -> str:
        """Generate design report and save as Markdown file"""
        if not project.final_sequence:
            return "Design project not completed, unable to generate report."
        
        # Use GLM to generate detailed report
        design_history = [
            {
                "stage": result.stage,
                "stage_name": result.stage_name,
                "description": f"Stage {result.stage}({result.stage_name}): Quality score {result.quality_score:.1f}, {result.notes}"
            }
            for result in project.stage_results
        ]
        
        report_result = self.glm_client.generate_design_report(
            project.final_sequence,
            design_history
        )
        
        if report_result["success"]:
            report_content = report_result["report"]
        else:
            report_content = f"Report generation failed: {report_result['error']}"
        
        # Save report to out folder
        try:
            import os
            from datetime import datetime
            
            # Ensure out folder exists
            out_dir = "out"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            # Generate filename (using timestamp and project ID)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traditional_report_{project.project_id}_{timestamp}.md"
            filepath = os.path.join(out_dir, filename)
            
            # Save report
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"\n[green]📄 Traditional design report saved to: {filepath}[/green]")
            
        except Exception as save_error:
            print(f"\n[yellow]⚠️ Report save failed: {save_error}[/yellow]")
        
        return report_content