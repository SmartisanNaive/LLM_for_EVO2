# 🧬 LLM4EVO2 Intelligent Sequence Design Platform

<div align="center">

![DNA-EVO2](https://img.shields.io/badge/DNA--EVO2-v2.1.0-blue?style=for-the-badge&logo=dna)
![AI-Powered](https://img.shields.io/badge/AI--Powered-Agent--Enhanced-green?style=for-the-badge&logo=robot)
![Python](https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

**🚀 Cell-free System Efficient Expression Regulatory Sequence Design Tool Based on NVIDIA EVO2-40B and Zhipu GLM-4.5**

*Integrates three-stage design workflow, LLM Agent intelligent optimization system, and natural language interaction interface to provide bioengineers with powerful DNA sequence design capabilities*

</div>

## 📋 Overview

This project implements a Python program based on the interaction between deep learning model EVO2 and large language model GLM-4.5-x, specifically designed for efficient expression regulatory sequence design in cell-free systems. Through a three-stage iterative optimization process, it achieves intelligent generation, validation, and optimization of DNA sequences, ultimately producing functional genetic regulatory elements.

## ✨ Features

- **Three-stage Iterative Design Process**: Unconstrained Exploration → Constrained Generation → Modular Validation
- **Dual Model Collaboration**: NVIDIA EVO2-40B + Zhipu GLM-4.5-x
- **LLM Agent Automatic Optimization System**: Intelligent Agent collaborative sequence optimization system
- **Professional Sequence Analysis**: GC content, functional element identification, secondary structure prediction
- **Cell-free System Optimization**: Specialized sequence optimization for cell-free expression systems
- **GFP Expression Experiment Demo**: Complete GFP expression regulatory sequence design workflow
- **Interactive User Interface**: Beautiful command-line interface and menu system based on Rich
- **FASTA Sequence Export**: Support for standard format sequence file export
- **Natural Language Interaction**: Multi-turn dialogue for sequence design requirements
- **Agent Intelligent Optimization**: Quality-driven automatic iteration and parameter adjustment

## 📦 Prerequisites

### Install uv (Python Package Manager)

**macOS:**
```bash
# Using Homebrew (recommended)
brew install uv

# Or using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Linux:**
```bash
# Using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

**Windows:**
```powershell
# Using PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

## 🚀 Installation

```bash
# Clone the project
git clone https://github.com/SmartisanNaive/LLM_for_EVO2.git
cd LLM_for_EVO2

# Create virtual environment and install dependencies using uv
uv sync

# Install project to virtual environment
uv pip install -e .
```

## ⚙️ Configuration

### API Setup

```bash
# Configure API keys
uv run evo2-designer setup

# Manage API configuration
uv run evo2-designer api-config

# Test API connection
uv run evo2-designer test
```

### Configuration Files

- **API Configuration**: `api_config.json` (project root)

Example configuration:
```json
{
  "nvidia_api_key": "your_nvidia_api_key_here",
  "glm_api_key": "your_glm_api_key_here",
  "evo2_base_url": "https://health.api.nvidia.com/v1/biology/nvidia/evo",
  "glm_base_url": "https://open.bigmodel.cn/api/paas/v4/"
}
```

## 🧬 Usage

### Interactive Interface (Recommended)

```bash
# Launch interactive menu
uv run python main.py
# or
uv run evo2-designer interactive
```

### Command Line Interface

```bash
# Basic commands
uv run evo2-designer setup                    # Initialize configuration
uv run evo2-designer test                     # Test API connection
uv run evo2-designer design --prompt "TAATACGACTCACTATAGGG" --length 99
uv run evo2-designer analyze --sequence "ATCGATCGATCG"
uv run evo2-designer list-projects            # View project list
uv run evo2-designer agent-config             # Configure Agent parameters
```

### GFP Expression Experiment

```bash
# Run GFP experiment demo
uv run evo2-designer gfp-demo --max-length 140 --target-length 120 --export
```

Designs DNA regulatory sequences for GFP protein expression in cell-free systems, including T7 promoter, 5'UTR with RBS, and start codon ATG.

### Programming Interface

```python
from evo2_sequence_designer import (
    Evo2Client, GLMClient, ThreeStageDesigner,
    DesignParameters
)

# Initialize clients
evo2_client = Evo2Client(Evo2Config(api_key="your_nvidia_api_key"))
glm_client = GLMClient(GLMConfig(api_key="your_glm_api_key"))

# Create designer
designer = ThreeStageDesigner(evo2_client, glm_client)

# Run design
parameters = DesignParameters(
    initial_prompt="TAATACGACTCACTATAGGG",
    target_length=99
)
project = designer.run_complete_design(parameters)
print(f"Final sequence: {project.final_sequence}")
```

## 📚 API Reference

### EVO2Client

```python
from evo2_sequence_designer.models.evo2_client import Evo2Client, Evo2Config

config = Evo2Config(api_key="your_nvidia_api_key")
client = Evo2Client(config)
response = client.generate_sequence(
    prompt="TAATACGACTCACTATAGGG",
    max_tokens=100,
    temperature=0.7
)
```

### GLMClient

```python
from evo2_sequence_designer.models.glm_client import GLMClient, GLMConfig

config = GLMConfig(api_key="your_glm_api_key")
client = GLMClient(config)
analysis = client.analyze_sequence(
    sequence="ATCGATCGATCG",
    analysis_type="optimization"
)
```

### SequenceAnalyzer

```python
from evo2_sequence_designer.analysis import SequenceAnalyzer

analyzer = SequenceAnalyzer()
analysis = analyzer.analyze_sequence("ATCGATCGATCG")
print(f"Quality score: {analysis.quality_score}")
```

## 🛠️ API Services

- **NVIDIA EVO2**: https://build.nvidia.com/arc/evo2-40b (DNA sequence generation)
- **Zhipu GLM**: https://open.bigmodel.cn/ (Biological analysis)

## 📁 Project Structure

```
evo2-sequence-designer/
├── main.py                     # Program entry
├── src/evo2_sequence_designer/
│   ├── main.py                 # Main module
│   ├── models/                 # Model interfaces
│   │   ├── evo2_client.py     # EVO2 client
│   │   └── glm_client.py      # GLM client
│   ├── analysis/              # Sequence analysis
│   ├── design/                # Design workflow
│   ├── agents/                # Agent system
│   └── demos/                 # Demo modules
├── api_config.json            # API configuration
└── pyproject.toml
```

## 🔧 Development

### Requirements
- Python 3.11+
- uv package manager
- NVIDIA API access
- Zhipu AI API access

### Dependencies
- `biopython`: Bioinformatics analysis
- `requests`: HTTP requests
- `zhipuai`: Zhipu AI SDK
- `rich`: CLI interface
- `typer`: Command line framework

### Extension Development

#### Adding New Agent
Extend `BaseAgent` class in `agents/` directory to implement custom optimization logic.

#### Custom Analysis
Extend `SequenceAnalyzer` in `analysis/` directory for new sequence analysis features.

#### New Models
Add new model clients in `models/` directory following the existing interface pattern.

#### Command Line Extensions
Add new commands using the `typer` framework in the main application.

## 📈 Performance

- API call caching
- Batch processing
- Intelligent retry mechanism
- Asynchronous requests

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Commit changes
4. Submit a Pull Request

## 📄 License

MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **NVIDIA**: EVO2-40B model API
- **Zhipu AI**: GLM-4.5-x model support
- **BioPython**: Bioinformatics tools

---

**Note**: Valid API keys required. Please comply with service terms.