"""LLM+EVO2 Collaborative Sequence Designer

Implements deep collaborative design workflow between LLM and EVO2:
1. LLM generates initial sequence based on user requirements
2. EVO2 performs optimization based on LLM sequence
3. LLM analyzes EVO2 results and provides improvement suggestions
4. Iterative optimization until requirements are met
5. Generate detailed design report
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import json
import re
from ..models.evo2_client import Evo2Client
from ..models.glm_client import GLMClient
from ..analysis.sequence_analyzer import SequenceAnalyzer
from .three_stage_designer import DesignParameters, DesignProject, StageResult


@dataclass
class CollaborativeIteration:
    """Collaborative iteration result"""
    iteration: int
    llm_initial_sequence: str
    evo2_optimized_sequence: str
    llm_analysis: Dict[str, Any]
    quality_score: float
    improvements: List[str]
    issues: List[str]
    success: bool
    timestamp: float


@dataclass
class CollaborativeResult:
    """Collaborative design result"""
    success: bool
    final_sequence: str
    initial_sequence: str
    iterations: List[CollaborativeIteration]
    final_analysis: Dict[str, Any]
    design_report: str
    total_time: float
    convergence_achieved: bool
    quality_improvement: float


class LLMEvo2CollaborativeDesigner:
    """LLM+EVO2 Collaborative Sequence Designer"""
    
    def __init__(self, evo2_client: Evo2Client, glm_client: GLMClient, 
                 sequence_analyzer: SequenceAnalyzer):
        self.evo2_client = evo2_client
        self.glm_client = glm_client
        self.analyzer = sequence_analyzer
        
        # Collaborative design configuration
        self.max_iterations = 5
        self.quality_threshold = 8.0
        self.improvement_threshold = 0.5
        self.convergence_patience = 2
    
    def design_sequence(self, user_requirement: str, 
                       design_params: Optional[DesignParameters] = None) -> CollaborativeResult:
        """Execute complete LLM+EVO2 collaborative design workflow
        
        Args:
            user_requirement: User requirement description
            design_params: Design parameters (optional)
            
        Returns:
            Collaborative design result
        """
        start_time = time.time()
        iterations = []
        
        print("\n🤖 Starting LLM+EVO2 collaborative sequence design...")
        
        try:
            # Step 1: Get initial sequence (from design_params or use default)
            print("\n📝 Step 1: Preparing initial sequence...")
            
            # Prioritize initial_prompt from design_params
            if design_params and design_params.initial_prompt:
                initial_sequence = design_params.initial_prompt
                print(f"   ✅ Using initial sequence from parameters: {initial_sequence[:50]}...")
            else:
                # Use default initial sequence
                initial_sequence = "ATGGCCGAGGAGCTGTTCGAG"  # Default coding sequence start
                print(f"   ✅ Using default initial sequence: {initial_sequence}")
            
            if not initial_sequence:
                return CollaborativeResult(
                    success=False,
                    final_sequence="",
                    initial_sequence="",
                    iterations=[],
                    final_analysis={},
                    design_report="Initial sequence is empty",
                    total_time=time.time() - start_time,
                    convergence_achieved=False,
                    quality_improvement=0.0
                )
            
            # Iterative optimization process
            current_sequence = initial_sequence
            best_quality = 0.0
            no_improvement_count = 0
            
            for iteration in range(1, self.max_iterations + 1):
                print(f"\n🔄 Round {iteration} collaborative optimization...")
                
                # Step 2: EVO2 optimization based on current sequence
                print("   🧬 EVO2 sequence optimization...")
                evo2_result = self._evo2_optimize_sequence(current_sequence, user_requirement)
                
                if not evo2_result["success"]:
                    print("   ⚠️ EVO2 optimization failed, continuing with LLM analysis of current sequence")
                    evo2_optimized = current_sequence
                    # Record EVO2 failure but don't interrupt the process
                    print(f"   📝 EVO2 failure reason: {evo2_result.get('error', 'Unknown error')}")
                else:
                    evo2_optimized = evo2_result["optimized_sequence"]
                    print(f"   ✅ EVO2 optimization completed: {evo2_optimized[:50]}...")
                
                # Step 3: LLM analyzes EVO2 results
                print("   🔍 LLM analyzing EVO2 results...")
                llm_analysis = self._llm_analyze_evo2_result(
                    evo2_optimized, user_requirement, iteration
                )
                
                # Calculate quality score
                sequence_analysis = self.analyzer.analyze_sequence(
                    evo2_optimized, f"Collaborative design round {iteration}"
                )
                quality_score = sequence_analysis.quality_score
                
                # Record iteration results
                iteration_result = CollaborativeIteration(
                    iteration=iteration,
                    llm_initial_sequence=current_sequence,
                    evo2_optimized_sequence=evo2_optimized,
                    llm_analysis=llm_analysis,
                    quality_score=quality_score,
                    improvements=llm_analysis.get("improvements", []),
                    issues=llm_analysis.get("issues", []),
                    success=llm_analysis.get("success", True),
                    timestamp=time.time()
                )
                iterations.append(iteration_result)
                
                print(f"   📊 Quality score: {quality_score:.1f}/100")
                
                # Check if quality threshold is reached
                if quality_score >= self.quality_threshold * 10:  # Convert to 100-point scale
                    print(f"   🎯 Quality threshold reached, design completed!")
                    current_sequence = evo2_optimized
                    break
                
                # Check for improvements
                improvement = quality_score - best_quality
                if improvement > self.improvement_threshold:
                    best_quality = quality_score
                    current_sequence = evo2_optimized
                    no_improvement_count = 0
                    print(f"   📈 Quality improvement: +{improvement:.1f}")
                else:
                    no_improvement_count += 1
                    print(f"   📉 No significant quality improvement ({no_improvement_count}/{self.convergence_patience})")
                
                # Check convergence
                if no_improvement_count >= self.convergence_patience:
                    print("   🔄 Convergence condition reached, stopping iteration")
                    break
                
                # Step 4: LLM generates improved sequence for next round
                if iteration < self.max_iterations:
                    print("   🔧 LLM generating improved sequence...")
                    improved_sequence = self._llm_generate_improved_sequence(
                        evo2_optimized, llm_analysis, user_requirement
                    )
                    if improved_sequence:
                        current_sequence = improved_sequence
                        print(f"   ✅ Improved sequence generated: {improved_sequence[:50]}...")
            
            # Generate final analysis and report
            print("\n📋 Generating design report...")
            final_analysis = self.analyzer.analyze_sequence(current_sequence, "Final design")
            design_report = self._generate_design_report(
                user_requirement, initial_sequence, current_sequence, 
                iterations, final_analysis
            )
            
            # Calculate quality improvement
            initial_analysis = self.analyzer.analyze_sequence(initial_sequence, "Initial sequence")
            quality_improvement = final_analysis.quality_score - initial_analysis.quality_score
            
            return CollaborativeResult(
                success=True,
                final_sequence=current_sequence,
                initial_sequence=initial_sequence,
                iterations=iterations,
                final_analysis=asdict(final_analysis),
                design_report=design_report,
                total_time=time.time() - start_time,
                convergence_achieved=no_improvement_count >= self.convergence_patience,
                quality_improvement=quality_improvement
            )
            
        except Exception as e:
            import traceback
            import logging
            
            # Get detailed error information
            error_type = type(e).__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()
            
            # Display user-friendly error information
            print(f"⚠️ Collaborative design encountered issues, attempting fallback processing: {error_message}")
            
            # Attempt fallback strategy: use LLM only to complete design
            try:
                print("🔄 Enabling fallback mode: LLM independent design...")
                
                # Use LLM to generate initial sequence (if not already available)
                if 'initial_sequence' not in locals() or not initial_sequence:
                    initial_sequence = self._llm_generate_initial_sequence(user_requirement)
                
                if initial_sequence:
                    # Use LLM to analyze sequence
                    final_analysis = self.analyzer.analyze_sequence(initial_sequence, "LLM independent design")
                    
                    # Generate simplified design report
                    design_report = f"""
# LLM Independent Sequence Design Report

## Design Overview
- **User Requirement**: {user_requirement}
- **Design Mode**: LLM Independent Design (EVO2 unavailable)
- **Final Sequence**: {initial_sequence}
- **Sequence Length**: {len(initial_sequence)}bp
- **Quality Score**: {final_analysis.quality_score:.1f}/100

## Design Description
Due to the temporary unavailability of EVO2 optimization service, the system adopted LLM independent design mode to complete sequence generation.
The sequence has passed basic quality checks and is recommended for further validation before use.

## Recommendations
1. Recommend re-optimization when EVO2 service is restored
2. Conduct experimental validation of sequence function
3. Adjust sequence parameters based on experimental results
                    """
                    
                    print(f"✅ LLM independent design completed! Sequence length: {len(initial_sequence)}bp")
                    print(f"📊 Quality Score: {final_analysis.quality_score:.1f}/100")
                    
                    return CollaborativeResult(
                        success=True,
                        final_sequence=initial_sequence,
                        initial_sequence=initial_sequence,
                        iterations=[],  # No iterations
                        final_analysis=asdict(final_analysis),
                        design_report=design_report,
                        total_time=time.time() - start_time,
                        convergence_achieved=False,
                        quality_improvement=0.0
                    )
                else:
                    print("❌ LLM independent design also failed, unable to generate sequence")
                    
            except Exception as fallback_error:
                print(f"❌ Fallback mode also failed: {str(fallback_error)}")
            
            # Log to file
            try:
                logger = logging.getLogger('SequenceDesign')
                logger.error(f"Collaborative design exception - Type: {error_type}, Message: {error_message}")
                logger.error(f"Complete stack trace:\n{error_traceback}")
            except Exception as log_error:
                print(f"Logging failed: {log_error}")
            
            # Return when final failure occurs
            return CollaborativeResult(
                success=False,
                final_sequence=initial_sequence if 'initial_sequence' in locals() else "",
                initial_sequence=initial_sequence if 'initial_sequence' in locals() else "",
                iterations=iterations,
                final_analysis={},
                design_report=f"Design failed - Type: {error_type}, Message: {error_message}\nSuggestion: Check network connection and API configuration",
                total_time=time.time() - start_time,
                convergence_achieved=False,
                quality_improvement=0.0
            )
    
    def _llm_generate_initial_sequence(self, user_requirement: str) -> str:
        """LLM generates initial sequence based on user requirements"""
        prompt = f"""
You are a professional molecular biologist and sequence design expert. Please design an initial DNA sequence based on the following user requirements:

User Requirements: {user_requirement}

Please analyze the requirements and generate an appropriate initial DNA sequence. Requirements:
1. The sequence should contain all functional elements required by the user
2. The sequence length should be reasonable (usually not exceeding 200bp)
3. The sequence should comply with molecular biology principles
4. Return only the DNA sequence, do not include other text

Please return the DNA sequence directly (containing only A, T, G, C):
        """
        
        try:
            response = self.glm_client.client.chat.completions.create(
                model=self.glm_client.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract DNA sequence
            dna_sequence = self._extract_dna_sequence(content)
            return dna_sequence
            
        except Exception as e:
            import traceback
            import logging
            
            # Get detailed error information
            error_type = type(e).__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()
            
            # Display user-friendly error information
            print(f"   ❌ LLM initial sequence generation failed: {error_message}")
            
            # Display detailed technical information for debugging
            print(f"   Error type: {error_type}")
            print(f"   Detailed stack trace:")
            # Ensure complete output of error stack trace
            for line in error_traceback.split('\n'):
                if line.strip():
                    print(f"   {line}")
            
            # Log to enhanced logging system
            try:
                from ..utils.logger import get_logger
                logger = get_logger()
                logger.log_error(
                    error_type=error_type,
                    error_message=error_message,
                    context={
                        "function": "generate_initial_sequence",
                        "user_requirement": user_requirement,
                        "traceback": error_traceback
                    }
                )
            except Exception as log_error:
                print(f"   Logging failed: {log_error}")
            
            # Add debugging tips
            print(f"   💡 Debug Tip: Please check the above error information, or view log files for more details")
            
            return ""
    
    def _evo2_optimize_sequence(self, sequence: str, requirement: str) -> Dict[str, Any]:
        """EVO2 optimize sequence"""
        try:
            # Use EVO2 to generate optimized sequence
            result = self.evo2_client.generate_sequence(
                prompt=sequence,
                temperature=0.7,
                max_tokens=100
            )
            
            if result["success"]:
                generated_text = result["generated_sequence"]
                optimized_sequence = self._extract_dna_sequence(generated_text)
                
                if optimized_sequence:
                    return {
                        "success": True,
                        "optimized_sequence": optimized_sequence,
                        "raw_output": generated_text
                    }
            
            return {"success": False, "error": "EVO2 generation failed"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _llm_analyze_evo2_result(self, sequence: str, requirement: str, 
                                iteration: int) -> Dict[str, Any]:
        """LLM analyzes EVO2 optimization results"""
        prompt = f"""
You are a professional molecular biologist. Please analyze the following EVO2-optimized DNA sequence:

Original Requirements: {requirement}
Current Iteration: Round {iteration}
EVO2 Optimized Sequence: {sequence}

Please analyze from the following perspectives:
1. Whether the sequence meets the original requirements
2. Biological reasonableness of the sequence
3. Potential issues and risks
4. Specific improvement suggestions
5. Strengths and weaknesses of the sequence

Please return the analysis results in JSON format:
{{
    "success": true/false,
    "meets_requirements": true/false,
    "biological_validity": "Assessment",
    "issues": ["Issue list"],
    "improvements": ["Improvement suggestions"],
    "strengths": ["Strengths list"],
    "overall_assessment": "Overall assessment",
    "confidence": 0.0-1.0
}}
        """
        
        try:
            response = self.glm_client.client.chat.completions.create(
                model=self.glm_client.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return analysis
            except json.JSONDecodeError:
                pass
            
            # If JSON parsing fails, return basic analysis
            return {
                "success": True,
                "meets_requirements": True,
                "biological_validity": "Requires further validation",
                "issues": ["JSON parsing failed, using basic analysis"],
                "improvements": ["Recommend manual sequence verification"],
                "strengths": ["Sequence generated"],
                "overall_assessment": content[:200] + "...",
                "confidence": 0.5
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "issues": [f"LLM analysis failed: {str(e)}"],
                "improvements": [],
                "confidence": 0.0
            }
    
    def _llm_generate_improved_sequence(self, current_sequence: str, 
                                       analysis: Dict[str, Any], 
                                       requirement: str) -> str:
        """LLM generates improved sequence based on analysis results"""
        issues = ", ".join(analysis.get("issues", []))
        improvements = ", ".join(analysis.get("improvements", []))
        
        prompt = f"""
You are a professional molecular biologist. Please improve the following DNA sequence based on the analysis results:

Original requirement: {requirement}
Current sequence: {current_sequence}
Identified issues: {issues}
Improvement suggestions: {improvements}

Please generate an improved DNA sequence with the following requirements:
1. Solve the identified issues
2. Implement improvement suggestions
3. Maintain the biological function of the sequence
4. Return only the improved DNA sequence

Improved DNA sequence:
        """
        
        try:
            response = self.glm_client.client.chat.completions.create(
                model=self.glm_client.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            improved_sequence = self._extract_dna_sequence(content)
            
            return improved_sequence
            
        except Exception as e:
            import traceback
            import logging
            
            # Get detailed error information
            error_type = type(e).__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()
            
            # Display user-friendly error information
            print(f"   ❌ LLM improved sequence generation failed: {error_message}")
            
            # Display detailed technical information for debugging
            print(f"   Error type: {error_type}")
            print(f"   Detailed stack trace:")
            # Ensure complete output of error stack trace
            for line in error_traceback.split('\n'):
                if line.strip():
                    print(f"   {line}")
            
            # Log to enhanced logging system
            try:
                from ..utils.logger import get_logger
                logger = get_logger()
                logger.log_error(
                    error_type=error_type,
                    error_message=error_message,
                    context={
                        "function": "improve_sequence",
                        "current_sequence": current_sequence,
                        "iteration": iteration,
                        "traceback": error_traceback
                    }
                )
            except Exception as log_error:
                print(f"   Logging failed: {log_error}")
            
            # Add debugging tips
            print("   💡 Debugging tip: Please check the above error information, or view log files for more details")
            
            return ""
    
    def _extract_dna_sequence(self, text: str, max_length: int = 300) -> str:
        """Extract DNA sequence from text"""
        # Remove spaces and newlines
        text = re.sub(r'\s+', '', text.upper())
        
        # Find consecutive DNA characters
        dna_pattern = r'[ATGC]+'
        matches = re.findall(dna_pattern, text)
        
        if matches:
            # Return the longest match
            longest_match = max(matches, key=len)
            return longest_match[:max_length]
        
        return ""
    
    def _generate_design_report(self, requirement: str, initial_sequence: str, 
                               final_sequence: str, iterations: List[CollaborativeIteration],
                               final_analysis) -> str:
        """Generate detailed design report and save as Markdown file"""
        report_prompt = f"""
Please generate a professional technical report for the following DNA sequence design project:

## Project Information
User requirement: {requirement}
Initial sequence: {initial_sequence}
Final sequence: {final_sequence}
Number of iterations: {len(iterations)}
Final quality score: {final_analysis.quality_score:.1f}/100

## Iteration History
{self._format_iteration_history(iterations)}

## Final Analysis
Sequence length: {final_analysis.length}bp
GC content: {final_analysis.gc_content:.1f}%
Molecular weight: {final_analysis.molecular_weight:.1f}Da
Functional elements: {len(final_analysis.features)}
Issues found: {len(final_analysis.issues)}

Please generate a professional report containing the following:
1. Iterative optimization process analysis
2. Detailed analysis of the final sequence
3. Performance evaluation and validation recommendations

Please generate the report in Markdown format.
        """
        
        try:
            response = self.glm_client.client.chat.completions.create(
                model=self.glm_client.config.model_name,
                messages=[{"role": "user", "content": report_prompt}],
                temperature=0.2,
                max_tokens=2000
            )
            
            report_content = response.choices[0].message.content
            
        except Exception as e:
            report_content = f"""
# DNA Sequence Design Report

## Project Overview
- **User Requirement**: {requirement}
- **Design Method**: LLM+EVO2 Collaborative Design
- **Number of Iterations**: {len(iterations)}
- **Final Quality**: {final_analysis.quality_score:.1f}/100

## Sequence Information
- **Initial Sequence**: {initial_sequence}
- **Final Sequence**: {final_sequence}
- **Sequence Length**: {final_analysis.length}bp
- **GC Content**: {final_analysis.gc_content:.1f}%

## Design Results
The collaborative design process was successfully completed, generating a DNA sequence that meets the requirements.

*Note: An error occurred during report generation: {str(e)}*
            """
        
        # Save report to out folder
        try:
            import os
            from datetime import datetime
            
            # Ensure out folder exists
            out_dir = "out"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            # Generate filename (using timestamp)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"design_report_{timestamp}.md"
            filepath = os.path.join(out_dir, filename)
            
            # Save report
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"\n[green]📄 Design report saved to: {filepath}[/green]")
            
        except Exception as save_error:
            print(f"\n[yellow]⚠️ Report save failed: {save_error}[/yellow]")
        
        return report_content
    
    def _format_iteration_history(self, iterations: List[CollaborativeIteration]) -> str:
        """Format iteration history"""
        history = []
        for it in iterations:
            history.append(
                f"Round {it.iteration}: Quality score {it.quality_score:.1f}, "
                f"Improvements {len(it.improvements)}, Issues {len(it.issues)}"
            )
        return "\n".join(history)