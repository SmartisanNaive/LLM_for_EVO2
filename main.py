#!/usr/bin/env python3
"""EVO2 sequence design platform - unified entry point"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# from rich.gradient import Gradient  # Remove incompatible import
from evo2_sequence_designer.main import app

def main():
    """Main entry function"""
    try:
        # If no command line arguments, start interactive interface
        if len(sys.argv) == 1:
            from evo2_sequence_designer.main import interactive
            interactive()
        else:
            # Otherwise use typer to handle command line arguments
            app()
    except KeyboardInterrupt:
        print("\n👋 Program exited")
    except Exception as e:
        print(f"❌ Program error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
