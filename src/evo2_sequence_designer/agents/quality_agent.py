"""Quality Assessment Agent - LLM-based intelligent sequence quality analysis"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
from ..models.glm_client import GLMClient
from ..analysis.sequence_analyzer import SequenceAnalyzer, SequenceAnalysisResult


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    overall_score: float  # Overall score (0-100)
    functional_score: float  # Functional score
    structural_score: float  # Structural score
    expression_score: float  # Expression efficiency score
    stability_score: float  # Stability score
    critical_issues: List[str]  # Critical issues
    minor_issues: List[str]  # Minor issues
    strengths: List[str]  # Strengths
    improvement_priority: List[str]  # Improvement priority


@dataclass
class QualityAssessment:
    """Quality assessment result"""
    sequence: str
    metrics: QualityMetrics
    detailed_analysis: str
    actionable_recommendations: List[str]
    confidence_level: float
    needs_iteration: bool
    iteration_strategy: Optional[str] = None


class QualityAssessmentAgent:
    """Quality Assessment Agent - Uses LLM for intelligent quality analysis"""
    
    def __init__(self, glm_client: GLMClient, sequence_analyzer: SequenceAnalyzer):
        self.glm_client = glm_client
        self.sequence_analyzer = sequence_analyzer
        self.quality_threshold = 75.0  # Quality threshold
        self.critical_threshold = 85.0  # Critical quality threshold
        
    def assess_sequence_quality(self, sequence: str, context: Dict[str, Any] = None) -> QualityAssessment:
        """Comprehensive sequence quality assessment
        
        Args:
            sequence: DNA sequence
            context: Design context information
            
        Returns:
            Quality assessment result
        """
        # 1. Basic analysis
        basic_analysis = self.sequence_analyzer.analyze_sequence(sequence)
        
        # 2. LLM deep analysis
        llm_analysis = self._perform_llm_analysis(sequence, basic_analysis, context)
        
        # 3. Comprehensive evaluation
        metrics = self._calculate_comprehensive_metrics(sequence, basic_analysis, llm_analysis)
        
        # 4. Generate recommendations
        recommendations = self._generate_actionable_recommendations(sequence, metrics, llm_analysis)
        
        # 5. Determine if iteration is needed
        needs_iteration, iteration_strategy = self._determine_iteration_need(metrics)
        
        return QualityAssessment(
            sequence=sequence,
            metrics=metrics,
            detailed_analysis=llm_analysis.get('detailed_analysis', ''),
            actionable_recommendations=recommendations,
            confidence_level=llm_analysis.get('confidence', 0.8),
            needs_iteration=needs_iteration,
            iteration_strategy=iteration_strategy
        )
    
    def _perform_llm_analysis(self, sequence: str, basic_analysis: SequenceAnalysisResult, 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform LLM deep analysis"""
        context_info = ""
        if context:
            context_info = f"""
Design Context:
- Target: {context.get('target', 'Not specified')}
- Application: {context.get('application', 'General')}
- Constraints: {context.get('constraints', 'None')}
"""
        
        features_info = "\n".join([
            f"- {f.name} ({f.start}-{f.end}): {f.description}"
            for f in basic_analysis.features
        ])
        
        issues_info = "\n".join([f"- {issue}" for issue in basic_analysis.issues])
        
        prompt = f"""As a synthetic biology and sequence design expert, please perform a deep quality analysis of the following DNA sequence:

Sequence: {sequence}
Length: {len(sequence)} bp
GC Content: {basic_analysis.gc_content:.1f}%
Quality Score: {basic_analysis.quality_score:.1f}

{context_info}

Detected functional elements:
{features_info}

Identified issues:
{issues_info}

Please provide the following analysis (reply in JSON format):
{{
  "functional_assessment": {{
    "score": 0-100,
    "analysis": "Functional analysis",
    "key_strengths": ["Strength 1", "Strength 2"],
    "critical_gaps": ["Gap 1", "Gap 2"]
  }},
  "structural_assessment": {{
    "score": 0-100,
    "secondary_structure_risk": "Low/Medium/High",
    "stability_prediction": "Analysis",
    "potential_issues": ["Issue 1", "Issue 2"]
  }},
  "expression_assessment": {{
    "score": 0-100,
    "transcription_efficiency": "Prediction",
    "translation_efficiency": "Prediction",
    "bottlenecks": ["Bottleneck 1", "Bottleneck 2"]
  }},
  "overall_assessment": {{
    "confidence": 0.0-1.0,
    "readiness_level": "Laboratory/Pilot/Production",
    "critical_issues": ["Critical issues"],
    "improvement_priority": ["Priority improvements"]
  }},
  "detailed_analysis": "Detailed professional analysis report"
}}

Please ensure the analysis is objective, accurate, and based on synthetic biology best practices."""
        
        try:
            response = self.glm_client.client.chat.completions.create(
                model=self.glm_client.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                # Extract JSON part
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                    return analysis_data
                else:
                    # If no JSON found, return text analysis
                    return {
                        "detailed_analysis": content,
                        "confidence": 0.7,
                        "functional_assessment": {"score": 70},
                        "structural_assessment": {"score": 70},
                        "expression_assessment": {"score": 70},
                        "overall_assessment": {"critical_issues": [], "improvement_priority": []}
                    }
            except json.JSONDecodeError:
                return {
                    "detailed_analysis": content,
                    "confidence": 0.6,
                    "functional_assessment": {"score": 60},
                    "structural_assessment": {"score": 60},
                    "expression_assessment": {"score": 60},
                    "overall_assessment": {"critical_issues": [], "improvement_priority": []}
                }
                
        except Exception as e:
            return {
                "error": f"LLM analysis failed: {str(e)}",
                "detailed_analysis": "Analysis failed, using basic assessment",
                "confidence": 0.5,
                "functional_assessment": {"score": basic_analysis.quality_score},
                "structural_assessment": {"score": basic_analysis.quality_score},
                "expression_assessment": {"score": basic_analysis.quality_score},
                "overall_assessment": {"critical_issues": basic_analysis.issues, "improvement_priority": []}
            }
    
    def _calculate_comprehensive_metrics(self, sequence: str, basic_analysis: SequenceAnalysisResult, 
                                       llm_analysis: Dict[str, Any]) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        # Extract LLM analysis results
        functional_score = llm_analysis.get('functional_assessment', {}).get('score', basic_analysis.quality_score)
        structural_score = llm_analysis.get('structural_assessment', {}).get('score', basic_analysis.quality_score)
        expression_score = llm_analysis.get('expression_assessment', {}).get('score', basic_analysis.quality_score)
        
        # Calculate stability score (based on GC content and length)
        gc_content = basic_analysis.gc_content
        stability_score = 100.0
        if gc_content < 30 or gc_content > 70:
            stability_score -= 20
        if len(sequence) > 140:
            stability_score -= 15
        
        # Calculate overall score (weighted average)
        overall_score = (
            functional_score * 0.3 +
            structural_score * 0.25 +
            expression_score * 0.25 +
            stability_score * 0.2
        )
        
        # Extract issues and strengths
        critical_issues = llm_analysis.get('overall_assessment', {}).get('critical_issues', [])
        critical_issues.extend([issue for issue in basic_analysis.issues if any(
            keyword in issue.lower() for keyword in ['missing', 'exceeds', 'not in']
        )])
        
        minor_issues = [issue for issue in basic_analysis.issues if issue not in critical_issues]
        
        strengths = llm_analysis.get('functional_assessment', {}).get('key_strengths', [])
        if basic_analysis.quality_score > 80:
            strengths.append("Good basic quality score")
        
        improvement_priority = llm_analysis.get('overall_assessment', {}).get('improvement_priority', [])
        
        return QualityMetrics(
            overall_score=max(0, min(100, overall_score)),
            functional_score=max(0, min(100, functional_score)),
            structural_score=max(0, min(100, structural_score)),
            expression_score=max(0, min(100, expression_score)),
            stability_score=max(0, min(100, stability_score)),
            critical_issues=list(set(critical_issues)),
            minor_issues=list(set(minor_issues)),
            strengths=list(set(strengths)),
            improvement_priority=improvement_priority
        )
    
    def _generate_actionable_recommendations(self, sequence: str, metrics: QualityMetrics, 
                                           llm_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable improvement recommendations"""
        recommendations = []
        
        # Generate recommendations based on critical issues
        for issue in metrics.critical_issues:
            if "missing promoter" in issue.lower():
                recommendations.append("Add T7 promoter sequence (TAATACGACTCACTATAGGG) to improve transcription efficiency")
            elif "missing rbs" in issue.lower() or "ribosome binding site" in issue.lower():
                recommendations.append("Add strong RBS sequence (AGGAGG) 5-9bp upstream of ATG")
            elif "missing start codon" in issue.lower():
                recommendations.append("Add ATG start codon at sequence end")
            elif "gc content" in issue.lower():
                if "below" in issue.lower() or "low" in issue.lower():
                    recommendations.append("Increase GC content to 40-60% range through synonymous codon substitution")
                else:
                    recommendations.append("Decrease GC content to 40-60% range through synonymous codon substitution")
            elif "length" in issue.lower():
                recommendations.append("Optimize sequence length, remove non-essential elements or use more compact design")
        
        # Generate recommendations based on scores
        if metrics.functional_score < 70:
            recommendations.append("Re-evaluate functional element selection and arrangement")
        
        if metrics.expression_score < 70:
            recommendations.append("Optimize RBS strength and 5'UTR structure to improve expression efficiency")
        
        if metrics.structural_score < 70:
            recommendations.append("Analyze and eliminate potential secondary structure issues")
        
        # Add LLM recommendations
        llm_bottlenecks = llm_analysis.get('expression_assessment', {}).get('bottlenecks', [])
        for bottleneck in llm_bottlenecks:
            recommendations.append(f"Address expression bottleneck: {bottleneck}")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _determine_iteration_need(self, metrics: QualityMetrics) -> Tuple[bool, Optional[str]]:
        """Determine if iterative optimization is needed"""
        needs_iteration = False
        strategy = None
        
        # Check if below quality threshold
        if metrics.overall_score < self.quality_threshold:
            needs_iteration = True
            
            # Determine iteration strategy
            if metrics.critical_issues:
                strategy = "critical_fix"  # Critical issue fix
            elif metrics.functional_score < 70:
                strategy = "functional_optimization"  # Functional optimization
            elif metrics.expression_score < 70:
                strategy = "expression_enhancement"  # Expression enhancement
            else:
                strategy = "general_improvement"  # General improvement
        
        # Check critical issues
        elif metrics.critical_issues:
            needs_iteration = True
            strategy = "critical_fix"
        
        return needs_iteration, strategy
    
    def compare_sequences(self, sequences: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compare quality of multiple sequences"""
        assessments = []
        
        for i, seq in enumerate(sequences):
            assessment = self.assess_sequence_quality(seq, context)
            assessments.append({
                'index': i,
                'sequence': seq,
                'assessment': assessment
            })
        
        # Sort by overall score
        assessments.sort(key=lambda x: x['assessment'].metrics.overall_score, reverse=True)
        
        # Generate comparison report
        best_seq = assessments[0]
        comparison_report = f"""
Best sequence (index {best_seq['index']}):
- Overall score: {best_seq['assessment'].metrics.overall_score:.1f}
- Functional score: {best_seq['assessment'].metrics.functional_score:.1f}
- Expression score: {best_seq['assessment'].metrics.expression_score:.1f}
- Critical issues count: {len(best_seq['assessment'].metrics.critical_issues)}

Ranking comparison:
"""
        
        for i, item in enumerate(assessments[:3]):  # Show top 3
            comparison_report += f"{i+1}. Sequence {item['index']}: {item['assessment'].metrics.overall_score:.1f} points\n"
        
        return {
            'assessments': assessments,
            'best_sequence': best_seq,
            'comparison_report': comparison_report,
            'recommendation': assessments[0]['assessment'].actionable_recommendations[0] if assessments[0]['assessment'].actionable_recommendations else "No specific recommendations"
        }