"""Logging system module - Records detailed logs of system operation"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

class SequenceDesignLogger:
    """Sequence design dedicated logger"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log file name (based on current time)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"sequence_design_{timestamp}.log"
        self.json_log_file = self.log_dir / f"sequence_design_{timestamp}.json"
        
        # Configure standard logging
        self.logger = logging.getLogger("SequenceDesign")
        self.logger.setLevel(logging.DEBUG)  # Changed to DEBUG level to record detailed information
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # JSON log record list
        self.json_logs = []
        
        # Record session start
        self.log_session_start()
    
    def log_session_start(self):
        """Record session start"""
        session_info = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "session_start",
            "message": "Sequence design session started"
        }
        self.logger.info("=== Sequence design session started ===")
        self._add_json_log(session_info)
    
    def log_design_start(self, parameters: Dict[str, Any]):
        """Record design start"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "design_start",
            "parameters": parameters,
            "message": f"Starting sequence design, initial prompt: {parameters.get('initial_prompt', 'N/A')}"
        }
        self.logger.info(f"🚀 Starting sequence design: {parameters}")
        self._add_json_log(log_entry)
    
    def log_stage_start(self, stage: str, stage_info: Dict[str, Any]):
        """Record stage start"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "stage_start",
            "stage": stage,
            "stage_info": stage_info,
            "message": f"Starting {stage} stage"
        }
        self.logger.info(f"📝 Starting {stage} stage: {stage_info}")
        self._add_json_log(log_entry)
    
    def log_iteration_start(self, iteration: int, strategy: str, parameters: Dict[str, Any]):
        """Record iteration start"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "iteration_start",
            "iteration": iteration,
            "strategy": strategy,
            "parameters": parameters,
            "message": f"Iteration {iteration} started, strategy: {strategy}"
        }
        self.logger.info(f"📊 Iteration {iteration} started, strategy: {strategy}, parameters: {parameters}")
        self._add_json_log(log_entry)
    
    def log_iteration_result(self, iteration: int, result: Dict[str, Any]):
        """Record iteration result"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "iteration_result",
            "iteration": iteration,
            "result": result,
            "message": f"Iteration {iteration} completed, score: {result.get('score', 'N/A')}"
        }
        
        if result.get('success', False):
            self.logger.info(f"✅ Iteration {iteration} successful: score {result.get('score', 'N/A')}, improvement {result.get('improvement', 'N/A')}")
        else:
            self.logger.warning(f"⚠️ Iteration {iteration} no improvement: {result.get('reason', 'Unknown reason')}")
        
        self._add_json_log(log_entry)
    
    def log_quality_assessment(self, sequence: str, assessment: Dict[str, Any]):
        """Record quality assessment"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "quality_assessment",
            "sequence_length": len(sequence),
            "assessment": assessment,
            "message": f"Quality assessment completed, overall score: {assessment.get('overall_score', 'N/A')}"
        }
        self.logger.info(f"🔍 Quality assessment: overall score {assessment.get('overall_score', 'N/A')}, functional {assessment.get('functional_score', 'N/A')}, expression efficiency {assessment.get('expression_score', 'N/A')}")
        self._add_json_log(log_entry)
    
    def log_agent_decision(self, agent_type: str, decision: Dict[str, Any]):
        """Record agent decision"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "agent_decision",
            "agent_type": agent_type,
            "decision": decision,
            "message": f"{agent_type} Agent decision: {decision.get('action', 'N/A')}"
        }
        self.logger.info(f"🤖 {agent_type} Agent decision: {decision}")
        self._add_json_log(log_entry)
    
    def log_optimization_result(self, final_result: Dict[str, Any]):
        """Record final optimization result"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "optimization_complete",
            "final_result": final_result,
            "message": f"Optimization completed, final score: {final_result.get('final_score', 'N/A')}"
        }
        self.logger.info(f"🎯 Optimization completed: final score {final_result.get('final_score', 'N/A')}, total iterations {final_result.get('total_iterations', 'N/A')}")
        self._add_json_log(log_entry)
    
    def log_llm_optimization_start(self, sequence: str, target_specs: Dict[str, Any], goals: List[str] = None):
        """Record LLM optimization start"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "llm_optimization_start",
            "sequence_length": len(sequence),
            "target_specs": target_specs,
            "optimization_goals": goals or [],
            "message": f"LLM optimization started, sequence length: {len(sequence)}bp"
        }
        self.logger.info(f"🧠 LLM optimization started: sequence length {len(sequence)}bp, goals: {goals or []}")
        self._add_json_log(log_entry)
    
    def log_llm_optimization_result(self, result_data: Dict[str, Any]):
        """Record LLM optimization result"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "llm_optimization_result",
            "result_data": result_data,
            "message": f"LLM optimization result: {result_data.get('status', 'N/A')}"
        }
        self.logger.info(f"🧠 LLM optimization result: {result_data}")
        self._add_json_log(log_entry)
    
    def log_llm_request(self, model_name: str, prompt: str, parameters: Dict[str, Any] = None):
        """Record LLM request input"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "llm_request",
            "model_name": model_name,
            "prompt": prompt,
            "parameters": parameters or {},
            "prompt_length": len(prompt),
            "message": f"LLM request: {model_name}, prompt length: {len(prompt)}"
        }
        self.logger.info(f"📤 LLM request [{model_name}]: prompt length {len(prompt)} characters")
        self.logger.debug(f"📤 LLM request details: {prompt[:200]}..." if len(prompt) > 200 else f"📤 LLM request details: {prompt}")
        self._add_json_log(log_entry)
    
    def log_llm_response(self, model_name: str, response: str, metadata: Dict[str, Any] = None):
        """Record LLM response output"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "llm_response",
            "model_name": model_name,
            "response": response,
            "metadata": metadata or {},
            "response_length": len(response),
            "message": f"LLM response: {model_name}, response length: {len(response)}"
        }
        self.logger.info(f"📥 LLM response [{model_name}]: response length {len(response)} characters")
        self.logger.debug(f"📥 LLM response details: {response[:200]}..." if len(response) > 200 else f"📥 LLM response details: {response}")
        self._add_json_log(log_entry)
    
    def log_evo2_request(self, sequence: str, parameters: Dict[str, Any] = None):
        """Record EVO2 request input"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "evo2_request",
            "sequence": sequence,
            "sequence_length": len(sequence),
            "parameters": parameters or {},
            "message": f"EVO2 request: sequence length {len(sequence)}bp"
        }
        self.logger.info(f"🧬 EVO2 request: sequence length {len(sequence)}bp")
        self.logger.debug(f"🧬 EVO2 input sequence: {sequence}")
        if parameters:
            self.logger.debug(f"🧬 EVO2 parameters: {parameters}")
        self._add_json_log(log_entry)
    
    def log_evo2_response(self, result: Dict[str, Any]):
        """Record EVO2 response output"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "evo2_response",
            "result": result,
            "message": f"EVO2 response: status {result.get('status', 'N/A')}"
        }
        self.logger.info(f"🧬 EVO2 response: {result.get('status', 'N/A')}")
        self.logger.debug(f"🧬 EVO2 response details: {result}")
        self._add_json_log(log_entry)
    
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Record error"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "message": f"Error: {error_type} - {error_message}"
        }
        self.logger.error(f"❌ {error_type}: {error_message}")
        if context:
            self.logger.error(f"Context: {context}")
        self._add_json_log(log_entry)
    
    def log_llm_optimization(self, sequence_before: str, sequence_after: str, 
                           optimization_details: Dict[str, Any]):
        """Record LLM optimization process"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "llm_optimization",
            "sequence_before_length": len(sequence_before),
            "sequence_after_length": len(sequence_after),
            "optimization_details": optimization_details,
            "message": f"LLM optimization completed, sequence length: {len(sequence_before)} -> {len(sequence_after)}"
        }
        self.logger.info(f"🧠 LLM optimization: {len(sequence_before)}bp -> {len(sequence_after)}bp, improvement: {optimization_details.get('improvement', 'N/A')}")
        self._add_json_log(log_entry)
    
    def _add_json_log(self, log_entry: Dict[str, Any]):
        """Add JSON format log"""
        self.json_logs.append(log_entry)
    
    def save_json_logs(self):
        """Save JSON format logs to file"""
        try:
            with open(self.json_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.json_logs, f, ensure_ascii=False, indent=2)
            self.logger.info(f"📄 JSON logs saved to: {self.json_log_file}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON logs: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        total_events = len(self.json_logs)
        event_types = {}
        
        for log in self.json_logs:
            event_type = log.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_events": total_events,
            "event_types": event_types,
            "session_duration": self._calculate_session_duration(),
            "log_files": {
                "text_log": str(self.log_file),
                "json_log": str(self.json_log_file)
            }
        }
    
    def _calculate_session_duration(self) -> str:
        """Calculate session duration"""
        if len(self.json_logs) < 2:
            return "0 minutes"
        
        start_time = datetime.fromisoformat(self.json_logs[0]['timestamp'])
        end_time = datetime.fromisoformat(self.json_logs[-1]['timestamp'])
        duration = end_time - start_time
        
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)
        
        return f"{minutes}m {seconds}s"
    
    def close(self):
        """Close logger and save final logs"""
        self.logger.info("=== Sequence design session ended ===")
        
        # Add session end log
        session_end = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "session_end",
            "summary": self.get_session_summary(),
            "message": "Sequence design session ended"
        }
        self._add_json_log(session_end)
        
        # Save JSON logs
        self.save_json_logs()
        
        # Close handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


# Global logger instance
_global_logger: Optional[SequenceDesignLogger] = None

def get_logger() -> SequenceDesignLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SequenceDesignLogger()
    return _global_logger

def setup_session_logger(session_type: str) -> str:
    """Setup session logger
    
    Args:
        session_type: Session type
        
    Returns:
        str: Session ID
    """
    session_id = f"{session_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = get_logger()
    logger.session_id = session_id
    return session_id


def close_logger():
    """Close global logger"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None