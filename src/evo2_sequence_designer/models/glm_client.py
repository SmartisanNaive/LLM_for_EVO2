"""GLM-4.5-x model client for intelligent analysis and decision support"""

from zhipuai import ZhipuAI
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from ..utils.logger import get_logger


@dataclass
class GLMConfig:
    """GLM model configuration"""
    api_key: str
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    model_name: str = "glm-4.5-x"
    temperature: float = 0.7
    max_tokens: int = 1024


class GLMClient:
    """GLM model client"""
    
    def __init__(self, config: GLMConfig):
        self.config = config
        self.client = ZhipuAI(api_key=config.api_key)
        self.logger = get_logger()
    
    def analyze_sequence_biology(self, sequence: str, stage: str = "") -> Dict[str, Any]:
        """Analyze the biological rationality of sequences
        
        Args:
            sequence: DNA sequence
            stage: Design stage information
            
        Returns:
            Analysis result dictionary
        """
        prompt = f"""As a synthetic biology expert, please analyze the biological rationality of the following DNA sequence:

Sequence: {sequence}
Design stage: {stage}

Please analyze from the following perspectives:
1. Sequence composition and GC content
2. Functional element identification (promoter, RBS, UTR, etc.)
3. Potential secondary structure issues
4. Expression potential in cell-free systems
5. Improvement suggestions

Please provide a detailed analysis report."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "success": True,
                "analysis": analysis,
                "sequence": sequence,
                "stage": stage
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"GLM analysis failed: {str(e)}",
                "sequence": sequence
            }
    
    def suggest_optimization(self, sequence: str, issues: List[str]) -> Dict[str, Any]:
        """Provide optimization suggestions based on discovered issues
        
        Args:
            sequence: Current sequence
            issues: List of discovered issues
            
        Returns:
            Optimization suggestion dictionary
        """
        issues_text = "\n".join([f"- {issue}" for issue in issues])
        
        prompt = f"""Based on the following DNA sequence and discovered issues, please provide specific optimization suggestions:

Current sequence: {sequence}

Discovered issues:
{issues_text}

Please provide:
1. Specific solutions for each issue
2. Recommended sequence modification strategies
3. Alternative functional element suggestions
4. Expected effects after optimization

Please give practical and actionable suggestions."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature ensures consistency of suggestions
                max_tokens=self.config.max_tokens
            )
            
            suggestions = response.choices[0].message.content
            
            return {
                "success": True,
                "suggestions": suggestions,
                "sequence": sequence,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Optimization suggestion generation failed: {str(e)}",
                "sequence": sequence
            }
    
    def evaluate_design_strategy(self, stage: int, previous_results: List[Dict]) -> Dict[str, Any]:
        """Evaluate design strategy and provide next step recommendations
        
        Args:
            stage: Current design stage (1, 2, 3)
            previous_results: Results from previous stages
            
        Returns:
            Strategy evaluation results
        """
        stage_names = {1: "Unconstrained exploration", 2: "Constrained generation", 3: "Modular validation"}
        current_stage = stage_names.get(stage, "Unknown stage")
        
        results_summary = "\n".join([
            f"Stage {i+1}: {result.get('summary', 'No results')}" 
            for i, result in enumerate(previous_results)
        ])
        
        prompt = f"""As a sequence design expert, please evaluate the current design strategy:

Current stage: {current_stage}
Previous results:
{results_summary}

Please provide:
1. Effectiveness evaluation of current strategy
2. Discovered patterns and trends
3. Best action plan for next steps
4. Potential risks and countermeasures
5. Success probability estimation

Please provide professional strategic recommendations."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=self.config.max_tokens
            )
            
            evaluation = response.choices[0].message.content
            
            return {
                "success": True,
                "evaluation": evaluation,
                "stage": stage,
                "stage_name": current_stage
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Strategy evaluation failed: {str(e)}",
                "stage": stage
            }
    
    def generate_design_report(self, final_sequence: str, design_history: List[Dict]) -> Dict[str, Any]:
        """Generate final design report
        
        Args:
            final_sequence: Final designed sequence
            design_history: Complete design history
            
        Returns:
            Design report
        """
        history_summary = "\n".join([
            f"Stage {i+1}: {entry.get('description', 'No description')}" 
            for i, entry in enumerate(design_history)
        ])
    
    def _parse_requirement_text(self, text: str) -> Dict[str, Any]:
        """Parse requirement analysis text content using keyword extraction and pattern matching
        
        Args:
            text: Text content returned by GLM
            
        Returns:
            Parsed structured data
        """
        import re
        
        # Default results
        result = {
            "sequence_type": "regulatory sequence",
            "target_length": 120,
            "function_requirements": ["transcriptional regulation"],
            "special_requirements": ["contains promoter and RBS"],
            "application_scenario": "cell-free expression system",
            "suggested_prompt": "TAATACGACTCACTATAGGG",
            "design_strategy": "intelligent design based on user requirements"
        }
        
        if not text or len(text.strip()) == 0:
            return result
        
        try:
            # 1. Sequence type extraction
            type_patterns = [
                r'sequence type[：:](.*?)(?=\n|$)',
                r'type[：:](.*?)(?=\n|$)',
                r'(promoter|coding sequence|regulatory sequence|terminator|enhancer)'
            ]
            for pattern in type_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result["sequence_type"] = match.group(1).strip() if match.group(1) else match.group(0)
                    break
            
            # 2. Target length extraction
            length_patterns = [
                r'target length[：:].*?(\d+)',
                r'length[：:].*?(\d+)',
                r'(\d+)\s*bp',
                r'(\d+)\s*bases?'
            ]
            for pattern in length_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        result["target_length"] = int(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
            
            # 3. Function requirements extraction
            function_patterns = [
                r'function requirements[：:](.*?)(?=\n|$)',
                r'function[：:](.*?)(?=\n|$)',
                r'(transcription|translation|regulation|expression|binding)'
            ]
            functions = []
            for pattern in function_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                functions.extend([m.strip() for m in matches if m.strip()])
            if functions:
                result["function_requirements"] = list(set(functions))[:3]  # Maximum 3 items
            
            # 4. Special requirements extraction
            special_patterns = [
                r'special requirements[：:](.*?)(?=\n|$)',
                r'special[：:](.*?)(?=\n|$)',
                r'(promoter|RBS|terminator|enhancer|repressor|ribosome binding site)'
            ]
            specials = []
            for pattern in special_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                specials.extend([m.strip() for m in matches if m.strip()])
            if specials:
                result["special_requirements"] = list(set(specials))[:3]  # Maximum 3 items
            
            # 5. Application scenario extraction
            scenario_patterns = [
                r'application scenario[：:](.*?)(?=\n|$)',
                r'scenario[：:](.*?)(?=\n|$)',
                r'(cell-free|in vitro|in vivo|prokaryotic|eukaryotic)'
            ]
            for pattern in scenario_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result["application_scenario"] = match.group(1).strip() if match.group(1) else match.group(0)
                    break
            
            # 6. Suggested initial sequence extraction
            prompt_patterns = [
                r'suggested.*?sequence[：:]\s*([ATCG]+)',
                r'initial sequence[：:]\s*([ATCG]+)',
                r'starting sequence[：:]\s*([ATCG]+)',
                r'([ATCG]{10,})'
            ]
            for pattern in prompt_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    sequence = match.group(1).upper()
                    if len(sequence) >= 10:  # At least 10 bases
                        result["suggested_prompt"] = sequence
                        break
            
            # 7. Design strategy extraction
            strategy_patterns = [
                r'design strategy[：:](.*?)(?=\n|$)',
                r'strategy[：:](.*?)(?=\n|$)',
                r'method[：:](.*?)(?=\n|$)'
            ]
            for pattern in strategy_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and match.group(1).strip():
                    result["design_strategy"] = match.group(1).strip()
                    break
            
        except Exception as e:
            # Log error during parsing but return default results
            self.logger.warning(f"Text parsing exception, using default results: {str(e)}")
        
        return result

    def analyze_sequence(self, sequence: str = "", analysis_type: str = "general", custom_prompt: str = "") -> Dict[str, Any]:
        """General sequence analysis method
        
        Args:
            sequence: DNA sequence (can be empty, for requirement analysis scenarios)
            analysis_type: Analysis type (general, requirement_analysis, etc.)
            custom_prompt: Custom analysis prompt
            
        Returns:
            Analysis result dictionary
        """
        if custom_prompt:
            prompt = custom_prompt
        elif analysis_type == "requirement_analysis":
            # Change to text format prompt, not forcing JSON requirement
            prompt = f"""Please analyze the following user requirements and extract key parameters:
            
User input: {sequence}
            
Please analyze and answer the following questions:
1. Sequence type: (such as regulatory sequence, coding sequence, promoter, etc.)
2. Target length: (recommended sequence length)
3. Function requirements: (main functional requirements)
4. Special requirements: (special design requirements)
5. Application scenario: (usage scenario)
6. Suggested initial sequence: (recommended starting sequence)
7. Design strategy: (design method recommendations)
            
Please answer with concise and clear text, one question per line."""
        else:
            prompt = f"""Please analyze the following DNA sequence:
            
Sequence: {sequence}
            
Please provide detailed biological analysis, including composition, functional prediction, and improvement suggestions."""
        
        try:
            # Log LLM request
            self.logger.log_llm_request(
                model_name=self.config.model_name,
                prompt=prompt,
                parameters={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "analysis_type": analysis_type
                }
            )
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            raw_content = response.choices[0].message.content
            
            # Log LLM response
            self.logger.log_llm_response(
                model_name=self.config.model_name,
                response=raw_content,
                metadata={
                    "analysis_type": analysis_type,
                    "sequence_length": len(sequence) if sequence else 0
                }
            )
            
            # If it's requirement analysis, use text parsing instead of JSON
            if analysis_type == "requirement_analysis":
                import re
                import json
                
                # Use keyword extraction and pattern matching to parse text content
                parsed_result = self._parse_requirement_text(raw_content)
                
                return {
                    "success": True,
                    "analysis": json.dumps(parsed_result, ensure_ascii=False),
                    "raw_content": raw_content,
                    "sequence": sequence,
                    "analysis_type": analysis_type,
                    "parsing_method": "text_extraction"  # Mark that text extraction method was used
                }
            
            # Non-requirement analysis, return raw content directly
            return {
                "success": True,
                "analysis": raw_content,
                "sequence": sequence,
                "analysis_type": analysis_type
            }
            
        except Exception as e:
            # Enhanced error handling, try to return basic results even if errors occur
            self.logger.error(f"GLM analysis exception: {str(e)}")
            
            if analysis_type == "requirement_analysis":
                # Provide default results for requirement analysis
                default_result = {
                    "sequence_type": "regulatory sequence",
                    "target_length": 120,
                    "function_requirements": ["transcriptional regulation"],
                    "special_requirements": ["contains promoter and RBS"],
                    "application_scenario": "cell-free expression system",
                    "suggested_prompt": "TAATACGACTCACTATAGGG",
                    "design_strategy": "intelligent design based on user requirements"
                }
                
                return {
                    "success": True,  # Mark as successful, using default results
                    "analysis": json.dumps(default_result, ensure_ascii=False),
                    "raw_content": f"Analysis exception, using default results: {str(e)}",
                    "sequence": sequence,
                    "analysis_type": analysis_type,
                    "fallback": True,
                    "error_handled": True
                }
            
            return {
                "success": False,
                "error": f"Sequence analysis failed: {str(e)}",
                "sequence": sequence
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test GLM API connection"""
        try:
            # Send a simple test request
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": "Hello, this is a connection test."}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            if response.choices and len(response.choices) > 0:
                return {"success": True, "message": "GLM API connection normal"}
            else:
                return {"success": False, "message": "GLM API response abnormal"}
                
        except Exception as e:
            return {"success": False, "message": f"GLM API connection failed: {str(e)}"}
        
        prompt = f"""Please generate a complete design report for the following DNA sequence design project:

Final sequence: {final_sequence}

Design process:
{history_summary}

The report should include:
1. Project overview and objectives
2. Design methods and strategies
3. Key findings from each stage
4. Functional analysis of the final sequence
5. Performance predictions and validation recommendations
6. Potential applications and improvement directions

Please generate a professional technical report."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2048
            )
            
            report = response.choices[0].message.content
            
            return {
                "success": True,
                "report": report,
                "final_sequence": final_sequence
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Report generation failed: {str(e)}",
                "final_sequence": final_sequence
            }