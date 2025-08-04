"""Evo2 model client for calling NVIDIA's evo2-40b model"""

import requests
import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from ..utils.logger import get_logger


@dataclass
class Evo2Config:
    """Evo2 model configuration"""
    api_key: str
    api_url: str = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"  # Complete API endpoint
    model_name: str = "arc/evo2-40b"
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 1
    max_retries: int = 3  # Maximum retry attempts
    retry_delay: float = 1.0  # Retry delay (seconds)


class Evo2Client:
    """Evo2 model client"""
    
    def __init__(self, config: Evo2Config):
        self.config = config
        self.logger = get_logger()
        
        # Use cloud API only, set Authorization header
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    def _get_api_url(self) -> str:
        """Get API URL"""
        return self.config.api_url
    

    
    def _make_request_with_retry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Request with retry mechanism"""
        url = self._get_api_url()
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=data, timeout=30)
                response.raise_for_status()
                
                # Handle response based on Content-Type
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return {"success": True, "response": response, "content_type": "json"}
                elif "application/zip" in content_type:
                    return {"success": True, "response": response, "content_type": "zip"}
                else:
                    return {"success": True, "response": response, "content_type": "unknown"}
                
            except requests.exceptions.SSLError as e:
                last_error = f"SSL connection error: {str(e)}"
                self.logger.log_error("SSL_ERROR", f"Attempt {attempt + 1}/{self.config.max_retries} failed: {last_error}")
                
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                self.logger.log_error("CONNECTION_ERROR", f"Attempt {attempt + 1}/{self.config.max_retries} failed: {last_error}")
                
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
                self.logger.log_error("REQUEST_ERROR", f"Attempt {attempt + 1}/{self.config.max_retries} failed: {last_error}")
            
            # If not the last attempt, wait and retry
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay)
        
        return {"success": False, "error": last_error}
    

    
    def generate_sequence(self, prompt: str, temperature: Optional[float] = None, 
                         max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate DNA sequence
        
        Args:
            prompt: Input DNA sequence as prompt
            temperature: Generation temperature, controls randomness
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing generation results
        """
        # Build request data - following reference code format
        data = {
            "sequence": prompt,
            "num_tokens": max_tokens or self.config.max_tokens,
            "top_k": self.config.top_k,
            "enable_sampled_probs": True
        }
        
        try:
            # Log EVO2 request
            self.logger.log_evo2_request(
                sequence=prompt,
                parameters={
                    "num_tokens": max_tokens or self.config.max_tokens,
                    "top_k": self.config.top_k,
                    "temperature": temperature or self.config.temperature
                }
            )
            
            # Send request with retry mechanism
            request_result = self._make_request_with_retry(data)
            
            if not request_result["success"]:
                return {
                    "success": False,
                    "error": request_result["error"],
                    "prompt": prompt
                }
            
            response = request_result["response"]
            content_type = request_result["content_type"]
            
            # Handle response based on Content-Type
            if content_type == "json":
                result = response.json()
                # Log EVO2 response
                self.logger.log_evo2_response(result)
                
                # Parse generated sequence
                generated_text = ""
                if "generated_sequence" in result:
                    generated_text = result["generated_sequence"]
                elif "sequence" in result:
                    generated_text = result["sequence"]
                
                return {
                    "success": True,
                    "generated_sequence": generated_text,
                    "sequences": [generated_text] if generated_text else [],
                    "prompt": prompt,
                    "parameters": {
                        "num_tokens": max_tokens or self.config.max_tokens,
                        "top_k": self.config.top_k
                    },
                    "raw_response": result
                }
            
            elif content_type == "zip":
                # Handle large ZIP format response
                self.logger.info("Received large ZIP format response")
                return {
                    "success": True,
                    "generated_sequence": "",
                    "sequences": [],
                    "prompt": prompt,
                    "zip_content": response.content,
                    "content_type": "zip",
                    "message": "Received ZIP format response, requires further processing"
                }
            
            else:
                # Unknown format
                self.logger.log_error("UNKNOWN_CONTENT_TYPE", f"Unknown response format: {content_type}")
                return {
                    "success": False,
                    "error": f"Unknown response format: {content_type}",
                    "prompt": prompt,
                    "raw_content": response.content[:200]  # Only log first 200 bytes
                }
            
        except Exception as e:
            self.logger.log_error("SEQUENCE_GENERATION_ERROR", f"Exception occurred during sequence generation: {str(e)}")
            return {
                "success": False,
                "error": f"Sequence generation exception: {str(e)}",
                "prompt": prompt
            }
    
    def validate_sequence(self, sequence: str, context: str = "") -> Dict[str, Any]:
        """Validate DNA sequence functionality
        
        Args:
            sequence: DNA sequence to validate
            context: Validation context information
            
        Returns:
            Dictionary containing validation results
        """
        # Simplified validation logic - check if sequence can generate continuation normally
        data = {
            "sequence": sequence,
            "num_tokens": 10,  # Generate only a few tokens for validation
            "top_k": 1,
            "enable_sampled_probs": True
        }
        
        try:
            # Send request with retry mechanism
            request_result = self._make_request_with_retry(data)
            
            if not request_result["success"]:
                return {
                    "success": False,
                    "error": f"Validation request failed: {request_result['error']}",
                    "sequence": sequence
                }
            
            response = request_result["response"]
            content_type = request_result["content_type"]
            
            if content_type == "json":
                result = response.json()
                
                # Parse generated sequence
                generated_text = ""
                if "generated_sequence" in result:
                    generated_text = result["generated_sequence"]
                elif "sequence" in result:
                    generated_text = result["sequence"]
                
                # Simple validation logic
                is_valid = len(generated_text) > 0 and all(c in 'ATCG' for c in generated_text.upper())
                
                return {
                    "success": True,
                    "is_valid": is_valid,
                    "analysis": f"Sequence validation passed, generated {len(generated_text)}bp continuation sequence",
                    "sequence": sequence,
                    "context": context,
                    "generated_continuation": generated_text
                }
            else:
                return {
                    "success": False,
                    "error": f"Validation response format error: {content_type}",
                    "sequence": sequence
                }
            
        except Exception as e:
            self.logger.log_error("SEQUENCE_VALIDATION_ERROR", f"Exception occurred during sequence validation: {str(e)}")
            return {
                "success": False,
                "error": f"Sequence validation failed: {str(e)}",
                "sequence": sequence
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection"""
        try:
            result = self.generate_sequence("ATCG", temperature=0.1, max_tokens=10)
            if result["success"]:
                return {"success": True, "message": "Evo2 API connection normal"}
            else:
                return {"success": False, "message": f"Connection test failed: {result['error']}"}
        except Exception as e:
            return {"success": False, "message": f"Connection test exception: {str(e)}"}