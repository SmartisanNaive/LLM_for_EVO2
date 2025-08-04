#!/usr/bin/env python3
"""
LLM4EVO2 startup script
Simplify project startup, automatically set Python path
"""

import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ['PYTHONPATH'] = str(project_root)

if __name__ == "__main__":
    # Import and run main program
    from src.evo2_sequence_designer.main import app
    
    # If no command line arguments provided, start interactive mode by default
    if len(sys.argv) == 1:
        sys.argv.append('interactive')
    
    app()