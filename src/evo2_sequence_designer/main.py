"""Main program entry - Cell-free system sequence design platform based on evo2 and GLM"""

import os
import json
import time
import random
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.rule import Rule
from rich.box import ROUNDED, DOUBLE, HEAVY
# from rich.gradient import Gradient  # Remove incompatible import
from rich.style import Style

from evo2_sequence_designer.models.evo2_client import Evo2Client, Evo2Config
from evo2_sequence_designer.models.glm_client import GLMClient, GLMConfig
from evo2_sequence_designer.design.three_stage_designer import ThreeStageDesigner, DesignParameters, DesignProject
from evo2_sequence_designer.design.llm_evo2_collaborative_designer import LLMEvo2CollaborativeDesigner, CollaborativeResult
from evo2_sequence_designer.analysis.sequence_analyzer import SequenceAnalyzer
# Remove GFP Demo related imports
from evo2_sequence_designer.utils.logger import get_logger, setup_session_logger

# Global logger instance
logger = get_logger()

app = typer.Typer(help="🧬 Cell-free system sequence design platform based on evo2 and GLM")
console = Console()

# ASCII art and visual effects
# LLM4EVO2 ASCII art title
LLM4EVO2_ASCII_TITLE = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ██╗     ██╗     ███╗   ███╗██╗  ██╗███████╗██╗   ██╗ ██████╗ ██████╗   ║
║    ██║     ██║     ████╗ ████║██║  ██║██╔════╝██║   ██║██╔═══██╗╚════██╗  ║
║    ██║     ██║     ██╔████╔██║███████║█████╗  ██║   ██║██║   ██║ █████╔╝  ║
║    ██║     ██║     ██║╚██╔╝██║╚════██║██╔══╝  ╚██╗ ██╔╝██║   ██║██╔═══╝   ║
║    ███████╗███████╗██║ ╚═╝ ██║     ██║███████╗ ╚████╔╝ ╚██████╔╝███████╗  ║
║    ╚══════╝╚══════╝╚═╝     ╚═╝     ╚═╝╚══════╝  ╚═══╝   ╚═════╝ ╚══════╝  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

# Startup prompt messages
STARTUP_MESSAGES = [
    "🧬 System initializing...",
    "🧬 Loading complete", 
    "🧬 Welcome to LLM4EVO2 sequence design platform"
]

def show_startup_animation():
    """Display LLM4EVO2 ASCII art title and startup information"""
    # Display ASCII art title
    console.print(f"[bold cyan]{LLM4EVO2_ASCII_TITLE}[/bold cyan]")
    
    # Display startup information
    console.print("\n[bold green]🧬 Intelligent DNA Sequence Design Platform[/bold green]")
    console.print("[yellow]Integrated NVIDIA EVO2-40B Large DNA Language Model + Zhipu GLM-4.5 Intelligent Analysis Engine[/yellow]")
    console.print("[blue]Three-stage Sequence Design Algorithm + Multi-Agent Collaborative Optimization System[/blue]\n")
    
    # Display loading status
    for message in STARTUP_MESSAGES:
        console.print(f"[cyan]{message}[/cyan]")
    
    console.print("\n" + "="*75)
    console.print("[bold blue]🚀 Ready to go, start your DNA design journey![/bold blue]")
    console.print("="*75 + "\n")

# Remove all animation functions

# Configuration file paths
CONFIG_DIR = Path.home() / ".evo2_designer"
CONFIG_FILE = CONFIG_DIR / "config.json"
API_CONFIG_FILE = Path(__file__).parent.parent.parent / "api_config.json"  # API configuration file in project root directory
PROJECTS_DIR = CONFIG_DIR / "projects"


def ensure_config_dir():
    """Ensure configuration directory exists"""
    CONFIG_DIR.mkdir(exist_ok=True)
    PROJECTS_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    """Load configuration file - prioritize reading API configuration file"""
    config = {}
    
    # First try to load API configuration file
    if API_CONFIG_FILE.exists():
        try:
            with open(API_CONFIG_FILE, 'r', encoding='utf-8') as f:
                api_config = json.load(f)
                config.update(api_config)
        except Exception as e:
            console.print(f"[yellow]⚠️ Failed to read API configuration file: {e}[/yellow]")
    
    # Then load main configuration file (for other settings)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                main_config = json.load(f)
                # API configuration takes priority, merge other configurations
                for key, value in main_config.items():
                    if key not in ['nvidia_api_key', 'glm_api_key', 'evo2_base_url', 'glm_base_url']:
                        config[key] = value
        except Exception as e:
            console.print(f"[yellow]⚠️ Failed to read main configuration file: {e}[/yellow]")
    
    return config


def save_config(config: dict):
    """Save configuration file"""
    ensure_config_dir()
    
    # Separate API configuration and other configurations
    api_keys = ['nvidia_api_key', 'glm_api_key', 'evo2_base_url', 'glm_base_url']
    api_config = {key: config.get(key, '') for key in api_keys if key in config}
    other_config = {key: value for key, value in config.items() if key not in api_keys}
    
    # Save API configuration to separate file
    if api_config:
        try:
            with open(API_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(api_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[red]❌ Failed to save API configuration file: {e}[/red]")
    
    # Save other configurations to main configuration file
    if other_config:
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(other_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[red]❌ Failed to save main configuration file: {e}[/red]")


def save_project(project: DesignProject):
    """Save project results"""
    ensure_config_dir()
    project_file = PROJECTS_DIR / f"{project.project_id}.json"
    
    # Convert to serializable format
    project_data = {
        "project_id": project.project_id,
        "parameters": {
            "initial_prompt": project.parameters.initial_prompt,
            "target_length": project.parameters.target_length,
            "max_length": project.parameters.max_length,
            "temperature_stage1": project.parameters.temperature_stage1,
            "temperature_stage2": project.parameters.temperature_stage2,
            "temperature_stage3": project.parameters.temperature_stage3,
            "max_iterations_per_stage": project.parameters.max_iterations_per_stage,
            "quality_threshold": project.parameters.quality_threshold
        },
        "stage_results": [
            {
                "stage": r.stage,
                "stage_name": r.stage_name,
                "success": r.success,
                "sequence": r.sequence,
                "quality_score": r.quality_score,
                "iteration": r.iteration,
                "timestamp": r.timestamp,
                "notes": r.notes,
                "issues": r.issues,
                "recommendations": r.recommendations
            }
            for r in project.stage_results
        ],
        "final_sequence": project.final_sequence,
        "status": project.status,
        "created_at": project.created_at,
        "completed_at": project.completed_at
    }
    
    with open(project_file, 'w', encoding='utf-8') as f:
        json.dump(project_data, f, indent=2, ensure_ascii=False)


@app.command()
def setup():
    """🔧 Initialize Configuration - Set API keys and parameters"""
    # Display concise startup prompt
    console.print("[cyan]🧬 System initializing...[/cyan]")
    
    # Create cool welcome interface
    welcome_layout = Layout()
    welcome_layout.split_column(
        Layout(Panel(
            Align.center(Text("🔧 System Initialization Configuration Wizard 🔧", style="bold cyan")),
            box=DOUBLE,
            border_style="cyan"
        ), size=3),
        Layout(Panel(
            "[bold green]Welcome to DNA-EVO2 Intelligent Design Platform![/bold green]\n\n"
            "[yellow]This platform integrates the following advanced technologies:[/yellow]\n"
            "🧬 NVIDIA EVO2-40B Large DNA Language Model\n"
            "🤖 Zhipu GLM-4.5 Intelligent Analysis Engine\n"
            "🔬 Three-stage Sequence Design Algorithm\n"
            "🚀 Multi-Agent Collaborative Optimization System\n\n"
            "[bold blue]Please configure API keys to start your DNA design journey![/bold blue]",
            title="🧬 Platform Introduction",
            border_style="green",
            box=ROUNDED
        ))
    )
    
    console.print(welcome_layout)
    
    config = load_config()
    
    # Configure NVIDIA evo2 API
    console.print("\n[bold blue]Configure NVIDIA evo2 API:[/bold blue]")
    nvidia_key = Prompt.ask("Please enter NVIDIA API key", 
                           default=config.get("nvidia_api_key", ""))
    
    # Configure GLM API
    console.print("\n[bold blue]Configure Zhipu GLM API:[/bold blue]")
    glm_key = Prompt.ask("Please enter GLM API key", 
                        default=config.get("glm_api_key", ""))
    
    # Save configuration
    config.update({
        "nvidia_api_key": nvidia_key,
        "glm_api_key": glm_key,
        "setup_completed": True
    })
    save_config(config)
    
    console.print("\n[green]✅ Configuration saved successfully![/green]")
    console.print(f"[blue]📁 API configuration saved to: {API_CONFIG_FILE}[/blue]")
    console.print(f"[blue]📁 Other configurations saved to: {CONFIG_FILE}[/blue]")
    
    # Test connections
    if Confirm.ask("Test API connections?"):
        test_connections()


@app.command()
def test():
    """🔍 Test API connections"""
    test_connections()


@app.command()
def api_config():
    """🔑 Manage API Configuration - View and modify API keys"""
    console.print(Panel.fit("🔑 API Configuration Management", title="Configuration Management", border_style="blue"))
    
    # Display current configuration status
    config = load_config()
    
    table = Table(title="Current API Configuration Status")
    table.add_column("Configuration Item", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Configuration File", style="blue")
    
    nvidia_status = "✅ Configured" if config.get("nvidia_api_key") else "❌ Not configured"
    glm_status = "✅ Configured" if config.get("glm_api_key") else "❌ Not configured"
    
    table.add_row("NVIDIA API Key", nvidia_status, str(API_CONFIG_FILE))
    table.add_row("GLM API Key", glm_status, str(API_CONFIG_FILE))
    
    console.print(table)
    
    # Operation selection
    console.print("\n[bold blue]Available Operations:[/bold blue]")
    console.print("1. Modify NVIDIA API key")
    console.print("2. Modify GLM API key")
    console.print("3. View configuration file paths")
    console.print("4. Test API connections")
    console.print("5. Exit")
    
    choice = Prompt.ask("Please select operation", choices=["1", "2", "3", "4", "5"], default="5")
    
    if choice == "1":
        new_key = Prompt.ask("Please enter new NVIDIA API key", 
                            default=config.get("nvidia_api_key", ""))
        config["nvidia_api_key"] = new_key
        save_config(config)
        console.print("[green]✅ NVIDIA API key updated[/green]")
        
    elif choice == "2":
        new_key = Prompt.ask("Please enter new GLM API key", 
                            default=config.get("glm_api_key", ""))
        config["glm_api_key"] = new_key
        save_config(config)
        console.print("[green]✅ GLM API key updated[/green]")
        
    elif choice == "3":
        console.print(f"\n[blue]📁 API configuration file: {API_CONFIG_FILE}[/blue]")
        console.print(f"[blue]📁 Main configuration file: {CONFIG_FILE}[/blue]")
        console.print(f"[blue]📁 Project directory: {PROJECTS_DIR}[/blue]")
        
    elif choice == "4":
        test_connections()
        
    elif choice == "5":
        console.print("[blue]👋 Exit configuration management[/blue]")
        return


def test_connections():
    """Test API connections"""
    config = load_config()
    
    if not config.get("setup_completed"):
        console.print("[red]❌ Please run 'setup' command first to configure API keys[/red]")
        return
    
    # Display connection test banner
    console.print(Panel(
        "[bold cyan]🔍 API Connection Diagnostic System[/bold cyan]\n\n"
        "Detecting connection status with external AI services...\n"
        "[yellow]⚡ High-speed network connection detection in progress[/yellow]",
        title="🌐 Connection Test",
        border_style="blue",
        box=HEAVY
    ))
    
    # Create connection status table
    status_table = Table(title="🔗 API Connection Status Monitor", box=ROUNDED)
    status_table.add_column("Service", style="cyan", width=20)
    status_table.add_column("Status", style="yellow", width=15)
    status_table.add_column("Latency", style="green", width=10)
    status_table.add_column("Details", style="blue")
    
    # Test evo2 connection
    console.print("\n🧬 [bold blue]Connecting to NVIDIA EVO2 service...[/bold blue]")
    evo2_config = Evo2Config(api_key=config["nvidia_api_key"])
    evo2_client = Evo2Client(evo2_config)
    
    start_time = time.time()
    with create_fancy_progress() as progress:
        task = progress.add_task("🧬 EVO2 Connection Test", total=100, status="Connecting...")
        for i in range(100):
            time.sleep(0.02)
            progress.update(task, advance=1, status=f"Testing... {i+1}%")
        evo2_result = evo2_client.test_connection()
    
    evo2_latency = f"{(time.time() - start_time)*1000:.0f}ms"
    
    if evo2_result["success"]:
        status_table.add_row("🧬 NVIDIA EVO2", "[green]● Online[/green]", evo2_latency, "DNA model ready")
        console.print("[green]✅ EVO2 API connection successful![/green]")
    else:
        status_table.add_row("🧬 NVIDIA EVO2", "[red]● Offline[/red]", "Timeout", f"Error: {evo2_result['message'][:30]}...")
        console.print(f"[red]❌ EVO2 API connection failed: {evo2_result['message']}[/red]")
    
    # Remove DNA helix animation
    
    # Test GLM connection
    console.print("\n🤖 [bold magenta]Connecting to Zhipu GLM service...[/bold magenta]")
    glm_config = GLMConfig(api_key=config["glm_api_key"])
    glm_client = GLMClient(glm_config)
    
    start_time = time.time()
    with create_fancy_progress() as progress:
        task = progress.add_task("🤖 GLM Connection Test", total=100, status="Connecting...")
        for i in range(100):
            time.sleep(0.015)
            progress.update(task, advance=1, status=f"Verifying... {i+1}%")
        glm_result = glm_client.test_connection()
    
    glm_latency = f"{(time.time() - start_time)*1000:.0f}ms"
    
    if glm_result["success"]:
        status_table.add_row("🤖 Zhipu GLM", "[green]● Online[/green]", glm_latency, "Intelligence engine ready")
        console.print("[green]✅ GLM API connection successful![/green]")
    else:
        status_table.add_row("🤖 Zhipu GLM", "[red]● Offline[/red]", "Timeout", f"Error: {glm_result['message'][:30]}...")
        console.print(f"[red]❌ GLM API connection failed: {glm_result['message']}[/red]")
    
    # Display final status table
    console.print("\n")
    console.print(status_table)
    
    # Connection summary
    all_connected = evo2_result["success"] and glm_result["success"]
    if all_connected:
        console.print(Panel(
            "[bold green]🎉 All API services connected successfully![/bold green]\n\n"
            "[cyan]System is ready, you can start DNA sequence design![/cyan]\n"
            "[yellow]💡 Recommend using 'agent-design' command to experience AI collaborative design[/yellow]",
            title="✅ Connection Successful",
            border_style="green",
            box=DOUBLE
        ))
    else:
        console.print(Panel(
            "[bold red]⚠️ Some API services have connection issues[/bold red]\n\n"
            "[yellow]Please check network connection and API key configuration[/yellow]\n"
            "[blue]💡 You can use 'api-config' command to reconfigure[/blue]",
            title="❌ Connection Issues",
            border_style="red",
            box=DOUBLE
        ))


@app.command()
def design(
    prompt: str = typer.Option("TAATACGACTCACTATAGGG", "--prompt", "-p", 
                              help="Initial DNA sequence prompt"),
    target_length: int = typer.Option(99, "--length", "-l", 
                                     help="Target sequence length"),
    project_id: Optional[str] = typer.Option(None, "--id", 
                                           help="Project ID (optional)"),
    enable_agent: bool = typer.Option(True, "--agent", "-a", 
                                     help="Enable Agent intelligent optimization")
):
    """🤖 Agent Intelligent Sequence Design - Automated optimization design process based on LLM Agent"""
    config = load_config()
    
    if not config.get("setup_completed"):
        console.print("[red]❌ Please run 'setup' command first to configure API keys[/red]")
        return
    
    # Display Agent design parameters
    console.print(Panel.fit(
        f"🤖 Agent Intelligent Sequence Design\n\n"
        f"Initial prompt: {prompt}\n"
        f"Target length: {target_length}bp\n"
        f"Project ID: {project_id or 'Auto-generated'}\n"
        f"Agent optimization: {'✅ Enabled' if enable_agent else '❌ Disabled'}\n\n"
        f"Agent will automatically execute:\n"
        f"• Quality assessment and feedback\n"
        f"• Iterative optimization strategy\n"
        f"• Intelligent parameter adjustment\n"
        f"• Real-time performance monitoring",
        title="🤖 Agent Intelligent Design System",
        border_style="blue"
    ))
    
    if not Confirm.ask("Confirm to start Agent intelligent design?"):
        return
    
    # Initialize client
    evo2_config = Evo2Config(api_key=config["nvidia_api_key"])
    evo2_client = Evo2Client(evo2_config)
    
    glm_config = GLMConfig(api_key=config["glm_api_key"])
    glm_client = GLMClient(glm_config)
    
    # Create Agent-enhanced designer
    designer = ThreeStageDesigner(
        evo2_client, glm_client, 
        enable_agent_optimization=enable_agent
    )
    
    # Set parameters
    parameters = DesignParameters(
        initial_prompt=prompt,
        target_length=target_length
    )
    
    # Run Agent intelligent design
    console.print("\n[bold blue]🤖 Starting Agent intelligent design process...[/bold blue]")
    
    try:
        project = designer.run_complete_design(parameters, project_id)
        
        # Save project
        save_project(project)
        
        # Display results
        display_design_results(project)
        
        # Display Agent optimization report
        if enable_agent and hasattr(designer, 'agent_coordinator'):
            console.print("\n[bold blue]🤖 Agent Optimization Report:[/bold blue]")
            agent_insights = designer.agent_coordinator.get_optimization_insights()
            if agent_insights:
                console.print(Panel(
                    str(agent_insights.get('summary', 'No optimization data')),
                    title="Agent Optimization Insights",
                    border_style="cyan"
                ))
        
        # Generate report
        if Confirm.ask("\nGenerate detailed design report?"):
            console.print("\n[bold blue]📊 Generating Agent design report...[/bold blue]")
            report = designer.generate_design_report(project)
            
            console.print("\n" + "="*80)
            console.print(Markdown(report))
            console.print("="*80)
            
            # Save report to out folder
            import os
            from datetime import datetime
            
            # Ensure out folder exists
            out_dir = "out"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            # Generate filename (using timestamp and project ID)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_report_{project.project_id}_{timestamp}.md"
            report_file = os.path.join(out_dir, filename)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            console.print(f"\n[green]📄 Agent report saved to: {report_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Agent design process failed: {str(e)}[/red]")
        if enable_agent:
            console.print("[yellow]💡 Tip: Try disabling Agent mode (--no-agent) to use traditional design process[/yellow]")


def display_design_results(project: DesignProject):
    """Display design results"""
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        f"Project ID: {project.project_id}\n"
        f"Status: {project.status}\n"
        f"Final sequence length: {len(project.final_sequence)}bp",
        title="🎯 Design Results Overview",
        border_style="green"
    ))
    
    # Display stage results
    table = Table(title="📊 Stage Results")
    table.add_column("Stage", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Quality Score", style="green")
    table.add_column("Sequence Length", style="yellow")
    table.add_column("Main Issues", style="red")
    
    for result in project.stage_results:
        issues_text = "; ".join(result.issues[:2]) if result.issues else "None"
        table.add_row(
            str(result.stage),
            result.stage_name,
            f"{result.quality_score:.1f}",
            f"{len(result.sequence)}bp",
            issues_text
        )
    
    console.print(table)
    
    # Display final sequence
    console.print("\n[bold blue]🧬 Final Optimized Sequence:[/bold blue]")
    sequence_display = Syntax(project.final_sequence, "text", 
                             theme="monokai", line_numbers=False)
    console.print(Panel(sequence_display, title="DNA Sequence", border_style="blue"))
    
    # Display sequence analysis
    if project.final_analysis:
        # Fix: Use SequenceAnalysisResult object attributes directly
        analysis = project.final_analysis
        console.print("\n[bold blue]📈 Sequence Analysis Results:[/bold blue]")
        
        analysis_table = Table()
        analysis_table.add_column("Metric", style="cyan")
        analysis_table.add_column("Value", style="green")
        
        analysis_table.add_row("Sequence Length", f"{analysis.length}bp")
        analysis_table.add_row("GC Content", f"{analysis.gc_content:.1f}%")
        analysis_table.add_row("Molecular Weight", f"{analysis.molecular_weight:.0f} Da")
        analysis_table.add_row("Quality Score", f"{analysis.quality_score:.1f}/100")
        analysis_table.add_row("Functional Elements", str(len(analysis.features)))
        
        console.print(analysis_table)


@app.command()
def analyze(sequence: str):
    """🔍 Analyze DNA sequence"""
    console.print(f"\n[bold blue]🔍 Analyzing sequence:[/bold blue] {sequence}")
    
    analyzer = SequenceAnalyzer()
    analysis = analyzer.analyze_sequence(sequence)
    
    table = Table(title="📊 Sequence Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Sequence Length", f"{analysis.length}bp")
    table.add_row("GC Content", f"{analysis.gc_content:.1f}%")
    table.add_row("Molecular Weight", f"{analysis.molecular_weight:.0f} Da")
    table.add_row("Quality Score", f"{analysis.quality_score:.1f}/100")
    table.add_row("Functional Elements", str(len(analysis.features)))
    
    console.print(table)
    
    # Display functional elements
    if analysis.features:
        console.print("\n[bold blue]🧬 Identified Functional Elements:[/bold blue]")
        features_table = Table()
        features_table.add_column("Name", style="cyan")
        features_table.add_column("Position", style="yellow")
        features_table.add_column("Sequence", style="green")
        features_table.add_column("Confidence", style="magenta")
        
        for feature in analysis.features:
            features_table.add_row(
                feature.name,
                f"{feature.start}-{feature.end}",
                feature.sequence,
                f"{feature.confidence:.2f}"
            )
        
        console.print(features_table)
    
    # Display issues and recommendations
    if analysis.issues:
        console.print("\n[bold red]⚠️  Issues Found:[/bold red]")
        for issue in analysis.issues:
            console.print(f"  • {issue}")
    
    if analysis.recommendations:
        console.print("\n[bold green]💡 Optimization Recommendations:[/bold green]")
        for rec in analysis.recommendations:
            console.print(f"  • {rec}")


# list_projects and show_project commands have been removed


@app.command()
def chat_design():
    """🤖 Intelligent Sequence Design - Agent-enhanced natural language interactive sequence design"""
    interactive_intelligent_design()


@app.command()
def agent_design(
    prompt: str = typer.Option("TAATACGACTCACTATAGGG", "--prompt", "-p", 
                              help="Initial sequence prompt"),
    target_length: int = typer.Option(120, "--length", "-l", 
                                     help="Target sequence length (bp)"),
    enable_agent: bool = typer.Option(True, "--agent", "-a", 
                                     help="Enable Agent intelligent optimization")
):
    """🤖 Agent Intelligent Design - Automated optimization design based on LLM Agent"""
    config = load_config()
    
    if not config.get("setup_completed"):
        console.print("[red]❌ Please configure API keys first[/red]")
        return
    
    # Display Agent system banner
    show_agent_banner()
    
    try:
        # Initialize clients
        evo2_config = Evo2Config(api_key=config["nvidia_api_key"])
        evo2_client = Evo2Client(evo2_config)
        
        glm_config = GLMConfig(api_key=config["glm_api_key"])
        glm_client = GLMClient(glm_config)
        
        # Simplified parameter display
        console.print(f"\n[cyan]🧬 Sequence parameters: {prompt} | {target_length}bp | Agent: {'✅' if enable_agent else '❌'}[/cyan]")
        
        # Confirm start design
        if not Confirm.ask("\n[bold blue]🚀 Start Agent intelligent design process?[/bold blue]"):
            console.print("[yellow]👋 Design cancelled[/yellow]")
            return
        
        params_table = Table()
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="white")
        
        params_table.add_row("Initial Prompt", prompt)
        params_table.add_row("Target Length", f"{target_length}bp")
        params_table.add_row("Agent Optimization", "✅ Enabled" if enable_agent else "❌ Disabled")
        
        console.print(params_table)
        
        # Build design parameters
        design_params = DesignParameters(
            initial_prompt=prompt,
            target_length=target_length
        )
        
        # Simplified process information
        if enable_agent:
            console.print("\n[cyan]🤖 Starting intelligent optimization process...[/cyan]")
        
        # Execute design
        console.print("\n[bold blue]🔬 Starting Agent design process...[/bold blue]")
        
        designer = ThreeStageDesigner(evo2_client, glm_client, enable_agent_optimization=enable_agent)
        project = designer.run_complete_design(design_params)
        
        # Simplified optimization results display
        if enable_agent and hasattr(project, 'agent_optimization_history') and project.agent_optimization_history:
            console.print(f"\n[green]🎯 Optimization completed: {len(project.agent_optimization_history)} iterations[/green]")
        
        # Display results
        display_design_results(project)
        
        # Save project
        save_project(project)
        
        console.print(f"\n[green]✅ Agent design completed! Project ID: {project.project_id}[/green]")
        
    except EOFError as e:
        console.print("\n[red]❌ Input stream error: Unable to read user input[/red]")
        console.print("[cyan]🔍 Error analysis:[/cyan]")
        console.print("   • Program running in non-interactive environment")
        console.print("   • Input redirected or piped")
        console.print("   • Terminal session abnormally interrupted")
        console.print("\n[blue]🛠️ Solutions:[/blue]")
        console.print("   1. Run directly in interactive terminal: python main.py agent-design")
        console.print("   2. Use command line arguments: python main.py design --prompt 'YOUR_PROMPT' --length 99")
        console.print("   3. Disable Agent mode: python main.py design --no-agent")
        console.print("   4. Check terminal environment: echo $TERM and tty commands")
        console.print("\n[yellow]💡 Tip: Recommend using non-interactive command line mode to avoid this issue[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ User interrupted operation[/yellow]")
        console.print("[green]👋 Agent design process safely exited[/green]")
    except Exception as e:
        console.print(f"\n[red]❌ Agent design process failed: {str(e)}[/red]")
        console.print(f"[dim]Error type: {type(e).__name__}[/dim]")
        
        # Provide specific suggestions based on error type
        if "API" in str(e) or "key" in str(e).lower():
            console.print("\n[cyan]🔑 API-related error suggestions:[/cyan]")
            console.print("   • Check API key configuration: python main.py api-config")
            console.print("   • Test API connection: python main.py test")
            console.print("   • Verify network connection status")
        elif "timeout" in str(e).lower() or "connection" in str(e).lower():
            console.print("\n[cyan]🌐 Network connection error suggestions:[/cyan]")
            console.print("   • Check network connection status")
            console.print("   • Try rerunning the program")
            console.print("   • Consider using VPN or proxy")
        else:
            console.print("\n[cyan]🔧 General troubleshooting suggestions:[/cyan]")
            console.print("   • Restart program: python main.py")
            console.print("   • Use traditional mode: python main.py design --no-agent")
            console.print("   • Check system resource usage")
            console.print("   • View detailed log files")
        
        import traceback
        console.print(f"\n[dim]Detailed error information:\n{traceback.format_exc()}[/dim]")


# GFP Demo functionality removed, focusing on sequence generation


@app.command()
def interactive():
    """🎮 Interactive Menu - Select different functions"""
    # Display ASCII art title on first entry
    first_run = True
    
    while True:
        console.clear()
        
        if first_run:
            # Display ASCII art title
            show_startup_animation()
            first_run = False
        else:
            # Display concise title in subsequent loops
            console.print("[bold blue]LLM4EVO2 - Intelligent DNA Sequence Design Platform[/bold blue]")
            
            # Display welcome interface
            title = Text("🧬 LLM4EVO2 Sequence Design Platform", style="bold blue")
            console.print(Panel.fit(title, border_style="blue"))
        
        # Display menu options
        menu_table = Table(show_header=False, box=None)
        menu_table.add_column("Option", style="cyan", width=4)
        menu_table.add_column("Function", style="white")
        menu_table.add_column("Description", style="dim")
        
        menu_table.add_row("1", "🔧 Initial Configuration", "Set API keys and parameters")
        menu_table.add_row("2", "🔑 API Configuration Management", "View and modify API keys")
        menu_table.add_row("3", "🔍 Test API Connection", "Verify evo2 and GLM API connections")
        menu_table.add_row("4", "🤖 Intelligent Sequence Design", "Agent-enhanced natural language interactive sequence design")
        menu_table.add_row("5", "🔍 Sequence Analysis", "Analyze DNA sequence characteristics")
        menu_table.add_row("6", "🤖 Agent Configuration Management", "Manage Agent intelligent optimization parameters")
        menu_table.add_row("0", "🚪 Exit Program", "")
        
        console.print("\n")
        console.print(menu_table)
        console.print("\n")
        
        try:
            choice = IntPrompt.ask("Please select function", choices=["0", "1", "2", "3", "4", "5", "6"])
            
            if choice == 0:
                console.print("[green]👋 Thank you for using LLM4EVO2 sequence design platform![/green]")
                break
            elif choice == 1:
                setup()
            elif choice == 2:
                api_config()
            elif choice == 3:
                test_connections()
            elif choice == 4:
                interactive_intelligent_design()
            elif choice == 5:
                interactive_analyze()
            elif choice == 6:
                manage_agent_config()
            
            if choice != 0:
                # Auto-save complete logs
                try:
                    logger.save_json_logs()
                    console.print("[dim]📄 Complete logs auto-saved[/dim]")
                except Exception as log_error:
                    console.print(f"[dim]⚠️ Log save failed: {log_error}[/dim]")
                
                console.print("\n[dim]Press Enter to continue...[/dim]")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[yellow]👋 Program exited[/yellow]")
                    break
                
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]⚠️ Input interruption detected[/yellow]")
            console.print("[cyan]💡 Possible causes:[/cyan]")
            console.print("   • Terminal input stream redirected or piped")
            console.print("   • Running in non-interactive environment")
            console.print("   • User pressed Ctrl+C or Ctrl+D")
            console.print("\n[blue]🔧 Solutions:[/blue]")
            console.print("   • Run program directly in interactive terminal")
            console.print("   • Use command line arguments instead of interactive input")
            console.print("   • Check terminal environment configuration")
            console.print("\n[green]👋 Program safely exited[/green]")
            break
        except Exception as e:
            console.print(f"[red]❌ Operation failed: {str(e)}[/red]")
            console.print(f"[dim]Error type: {type(e).__name__}[/dim]")
            console.print("\n[dim]Press Enter to continue...[/dim]")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]👋 Program exited[/yellow]")
                break


def interactive_design():
    """Interactive Agent intelligent design"""
    config = load_config()
    
    if not config.get("setup_completed"):
        console.print("[red]❌ Please configure API keys first[/red]")
        return
    
    console.print("\n[bold blue]🤖 Agent Intelligent Sequence Design[/bold blue]")
    
    # Get design parameters
    prompt = Prompt.ask("Initial DNA sequence prompt", default="TAATACGACTCACTATAGGG")
    target_length = IntPrompt.ask("Target sequence length (bp)", default=99)
    project_id = Prompt.ask("Project ID (optional, leave blank for auto-generation)", default="")
    enable_agent = Confirm.ask("Enable Agent intelligent optimization?", default=True)
    
    if not project_id:
        project_id = None
    
    # Set up session logger
    session_id = setup_session_logger("agent_design")
    logger = get_logger()
    
    # Log session start
    logger.log_session_start()
    
    # Log design parameters
    logger.log_design_start({
        'session_type': 'agent_design',
        'initial_prompt': prompt,
        'target_length': target_length,
        'project_id': project_id,
        'enable_agent': enable_agent
    })
    
    console.print(f"\n[dim]📝 Session ID: {session_id}[/dim]")
    console.print(f"[dim]📁 Log file: logs/{session_id}.log[/dim]")
    
    # Simplified feature description
    if enable_agent:
        console.print("[cyan]🤖 Intelligent optimization mode enabled[/cyan]")
    
    # Run Agent intelligent design
    design(prompt, target_length, project_id, enable_agent)


def interactive_intelligent_design():
    """Interactive intelligent sequence design - Integrating Agent optimization and natural language interaction"""
    config = load_config()
    
    if not config.get("setup_completed"):
        console.print("[red]❌ Please configure API keys first[/red]")
        return
    
    # Display intelligent design banner
    console.print(Panel(
        "[bold cyan]🧠 Intelligent Sequence Design System[/bold cyan]\n\n"
        "[green]✨ Natural Language Driven DNA Sequence Design[/green]\n"
        "[yellow]🤖 AI Agent Collaborative Optimization Technology[/yellow]\n"
        "[blue]🔬 Three-Stage Progressive Design Process[/blue]\n\n"
        "[bold magenta]Simply describe your requirements in natural language, AI will design optimal sequences for you![/bold magenta]",
        title="🧬 Intelligent Design Wizard",
        border_style="cyan",
        box=DOUBLE
    ))
    
    # Create requirements input interface
    console.print(Panel(
        "[bold yellow]💭 Please describe your sequence design requirements in detail[/bold yellow]\n\n"
        "[dim]Examples:\n"
        "• Design a promoter sequence for GFP expression\n"
        "• Need a 120bp T7 promoter regulatory sequence\n"
        "• Design efficient expression regulatory elements for cell-free systems[/dim]",
        title="📝 Requirements Description",
        border_style="yellow",
        box=ROUNDED
    ))
    
    # Get user requirements
    user_requirement = Prompt.ask("\n[bold blue]🎯 Your design requirements[/bold blue]")
    
    if not user_requirement.strip():
        console.print("[red]❌ Requirements description cannot be empty[/red]")
        return
    
    # Enable Agent optimization by default
    enable_agent = True
    console.print("\n[bold cyan]🤖 Using LLM+EVO2 collaborative design mode[/bold cyan]")
    
    try:
        # Initialize client
        evo2_config = Evo2Config(api_key=config["nvidia_api_key"])
        evo2_client = Evo2Client(evo2_config)
        
        glm_config = GLMConfig(api_key=config["glm_api_key"])
        glm_client = GLMClient(glm_config)
        
        console.print("\n[bold blue]🤖 Analyzing your requirements...[/bold blue]")
        
        # Use GLM to analyze user requirements
        analysis_prompt = f"""
        Please analyze the following user's DNA sequence design requirements and extract key parameters:
        
        User requirements: {user_requirement}
        
        Please extract the following information (if not explicitly stated by user, provide reasonable defaults):
        1. Sequence type (promoter, coding sequence, regulatory element, etc.)
        2. Target length (bp)
        3. Functional requirements (transcription, translation, regulation, etc.)
        4. Special requirements (GC content, restriction enzyme sites, etc.)
        5. Application scenario (in vitro expression, cell transfection, etc.)
        
        Please return analysis results in JSON format:
        {{
            "sequence_type": "sequence type",
            "target_length": target_length_number,
            "function_requirements": ["functional requirements list"],
            "special_requirements": ["special requirements list"],
            "application_scenario": "application scenario",
            "suggested_prompt": "suggested initial prompt sequence",
            "design_strategy": "design strategy description"
        }}
        """
        
        # Call GLM analysis
        analysis_result = glm_client.analyze_sequence(
            sequence=user_requirement,  # Pass user requirements instead of empty string
            analysis_type="requirement_analysis",
            custom_prompt=analysis_prompt
        )
        
        if not analysis_result.get("success"):
            console.print(f"[red]❌ Requirements analysis failed: {analysis_result.get('error', 'Unknown error')}[/red]")
            return
        
        # Add debug information
        if analysis_result.get("raw_content"):
            console.print("[dim]🔍 GLM raw response content:[/dim]")
            console.print(f"[dim]{analysis_result['raw_content'][:200]}...[/dim]")
        
        # Parse analysis results
        try:
            import json
            requirements = json.loads(analysis_result["analysis"])
            
            # Show if default results were used
            if analysis_result.get("fallback"):
                console.print("[yellow]⚠️ Using default analysis results (GLM response format error)[/yellow]")
                
        except json.JSONDecodeError as e:
            console.print(f"[red]❌ Requirements analysis result format error: {str(e)}[/red]")
            console.print(f"[dim]Raw content: {analysis_result.get('analysis', 'N/A')}[/dim]")
            return
        
        # Display analysis results
        console.print("\n[bold green]📋 Requirements Analysis Results:[/bold green]")
        
        req_table = Table()
        req_table.add_column("Item", style="cyan")
        req_table.add_column("Content", style="white")
        
        req_table.add_row("Design Target", requirements.get("sequence_type", "Unspecified"))
        req_table.add_row("Application Scenario", requirements.get("application_scenario", "General"))
        req_table.add_row("Sequence Type", requirements.get("sequence_type", "Regulatory sequence"))
        req_table.add_row("Length Requirement", f"{requirements.get('target_length', '120')}bp")
        req_table.add_row("Special Requirements", ", ".join(requirements.get("special_requirements", ["None"])))
        req_table.add_row("Agent Optimization", "✅ Enabled" if enable_agent else "❌ Disabled")
        
        console.print(req_table)
        
        # Confirm whether to continue
        if not Confirm.ask("\nProceed with sequence design based on this analysis?"):
            return
        
        # Build design parameters
        prompt = requirements.get("suggested_prompt", "TAATACGACTCACTATAGGG")
        target_length = requirements.get("target_length", 120)
        
        design_params = DesignParameters(
            initial_prompt=prompt,
            target_length=target_length
        )
        
        # Display design process start
        console.print("\n[bold blue]🔬 Starting intelligent sequence design process...[/bold blue]")
        
        # Simplified process information
        console.print("\n[bold blue]🔬 Launching intelligent sequence design...[/bold blue]")
        
        # Initialize sequence analyzer
        analyzer = SequenceAnalyzer()
        
        if enable_agent:
            # Use new LLM+EVO2 collaborative designer
            console.print("\n[bold cyan]🤖 Launching LLM+EVO2 collaborative design process...[/bold cyan]")
            
            collaborative_designer = LLMEvo2CollaborativeDesigner(
                evo2_client=evo2_client,
                glm_client=glm_client,
                sequence_analyzer=analyzer
            )
            
            # Execute collaborative design
            collaborative_result = collaborative_designer.design_sequence(
                user_requirement=user_requirement,
                design_params=design_params
            )
            
            if collaborative_result.success:
                # Display iterative optimization process analysis
                console.print("\n[bold cyan]📊 Iterative Optimization Process Analysis:[/bold cyan]")
                
                if collaborative_result.iterations:
                    collab_table = Table()
                    collab_table.add_column("Iteration Round", style="cyan")
                    collab_table.add_column("Quality Score", style="green")
                    collab_table.add_column("Improvement Strategy", style="yellow")
                    
                    for iteration in collaborative_result.iterations:
                        collab_table.add_row(
                            f"Round {iteration.iteration}",
                            f"{iteration.quality_score:.1f}/100",
                            f"{len(iteration.improvements)} improvements"
                        )
                    
                    console.print(collab_table)
                    
                    # Display design goal achievement
                    console.print(f"\n[bold green]🎯 Design Goal: Quality improvement +{collaborative_result.quality_improvement:.1f} points[/bold green]")
                
                # Display simplified design report
                console.print("\n[bold green]📋 Design Summary:[/bold green]")
                if collaborative_result.design_report:
                    # Only display key parts of the report
                    report_lines = collaborative_result.design_report.split('\n')
                    key_sections = []
                    for line in report_lines[:15]:  # Only take first 15 lines
                        if any(keyword in line.lower() for keyword in ['design goal', 'optimization', 'quality', 'sequence features']):
                            key_sections.append(line)
                    if key_sections:
                        console.print('\n'.join(key_sections[:5]))  # Display at most 5 lines of key information
                    else:
                        console.print("[dim]Design completed, detailed report saved[/dim]")
                
                # Create project object to save results
                from .models.project import Project, StageResult
                import uuid
                
                project = Project(
                    project_id=str(uuid.uuid4())[:8],
                    parameters=design_params
                )
                
                # Add collaborative design results
                final_stage = StageResult(
                    stage=3,
                    stage_name="LLM+EVO2 Collaborative Design",
                    success=True,
                    sequence=collaborative_result.final_sequence,
                    quality_score=collaborative_result.final_analysis.get('quality_score', 0),
                    iteration=len(collaborative_result.iterations),
                    timestamp=time.time(),
                    notes=f"Collaborative design completed, quality improvement: +{collaborative_result.quality_improvement:.1f} points"
                )
                
                project.stage_results.append(final_stage)
                project.final_sequence = collaborative_result.final_sequence
                project.status = "completed"
                project.completed_at = time.time()
                
                # Display final sequence
                console.print("\n[bold green]🧬 Final Sequence:[/bold green]")
                console.print(f"[green]{collaborative_result.final_sequence}[/green]")
                console.print(f"[dim]Length: {len(collaborative_result.final_sequence)}bp[/dim]")
                
            else:
                # Display detailed error information
                console.print(f"[red]❌ Intelligent design failed[/red]")
                
                # Display detailed error report
                if hasattr(collaborative_result, 'design_report') and collaborative_result.design_report:
                    console.print(f"[yellow]📋 Detailed error information:[/yellow]")
                    console.print(f"[red]{collaborative_result.design_report}[/red]")
                
                # Display other available debug information
                if hasattr(collaborative_result, 'iterations') and collaborative_result.iterations:
                    console.print(f"[dim]🔄 Completed iterations: {len(collaborative_result.iterations)}[/dim]")
                
                if hasattr(collaborative_result, 'total_time'):
                    console.print(f"[dim]⏱️ Total time: {collaborative_result.total_time:.2f} seconds[/dim]")
                
                console.print(f"[yellow]💡 Debugging suggestions:[/yellow]")
                console.print(f"[yellow]  - Check GLM API quota and connection status[/yellow]")
                console.print(f"[yellow]  - Verify input parameter validity[/yellow]")
                console.print(f"[yellow]  - Check complete log files for more information[/yellow]")
                return
        
        # Save project
        save_project(project)
        
        console.print(f"\n[green]✅ Intelligent design completed! Project ID: {project.project_id}[/green]")
        
    except Exception as e:
        console.print(f"[red]WARN: Issues may exist during the process, please check logs in production environment[/red]")


def interactive_natural_language_design():
    """Natural language interactive sequence design"""
    config = load_config()
    
    if not config.get("setup_completed"):
        console.print("[red]❌ Please configure API keys first[/red]")
        return
    
    console.print("\n[bold blue]💬 Natural Language Sequence Design[/bold blue]")
    
    # Display feature introduction
    console.print(Panel.fit(
        "🧬 Natural Language Sequence Design Features:\n\n"
        "• Describe your sequence requirements through conversation\n"
        "• AI understands and converts to design parameters\n"
        "• Intelligently generates DNA sequences meeting requirements\n"
        "• Provides detailed sequence analysis reports\n"
        "• Supports multi-round conversation optimization design",
        title="💬 Intelligent Conversation Design",
        border_style="cyan"
    ))
    
    # Initialize client
    evo2_config = Evo2Config(api_key=config["nvidia_api_key"])
    evo2_client = Evo2Client(evo2_config)
    
    glm_config = GLMConfig(api_key=config["glm_api_key"])
    glm_client = GLMClient(glm_config)
    
    # Start natural language interaction
    console.print("\n[bold green]🤖 AI assistant is ready, please describe your sequence design requirements:[/bold green]")
    
    while True:
        try:
            # Get user requirement description
            try:
                user_input = Prompt.ask("\n[cyan]Your requirements[/cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[green]👋 Thank you for using natural language sequence design![/green]")
                break
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("[green]👋 Thank you for using natural language sequence design![/green]")
                break
            
            # Use GLM to analyze user requirements
            console.print("\n[yellow]🤖 AI is understanding your requirements...[/yellow]")
            
            # Build analysis prompt
            analysis_prompt = f"""
            Please analyze the following user's DNA sequence design requirements and extract key parameters:
            
            User requirements: {user_input}
            
            Please extract the following information (if not explicitly stated by user, provide reasonable defaults):
            1. Sequence type (promoter, coding sequence, regulatory element, etc.)
            2. Target length (bp)
            3. Functional requirements (transcription, translation, regulation, etc.)
            4. Special requirements (GC content, restriction enzyme sites, etc.)
            5. Application scenario (in vitro expression, cell transfection, etc.)
            
            Please return analysis results in JSON format:
            {{
                "sequence_type": "sequence type",
                "target_length": target_length_number,
                "function_requirements": ["functional requirements list"],
                "special_requirements": ["special requirements list"],
                "application_scenario": "application scenario",
                "suggested_prompt": "suggested initial prompt sequence",
                "design_strategy": "design strategy description"
            }}
            """
            
            # Call GLM analysis
            analysis_result = glm_client.analyze_sequence(
                sequence=user_input,  # Pass user input instead of empty string
                analysis_type="requirement_analysis",
                custom_prompt=analysis_prompt
            )
            
            if analysis_result["success"]:
                # Add debug information
                if analysis_result.get("raw_content"):
                    console.print("[dim]🔍 GLM raw response content:[/dim]")
                    console.print(f"[dim]{analysis_result['raw_content'][:200]}...[/dim]")
                
                try:
                    # Parse JSON returned by GLM
                    import json
                    requirements = json.loads(analysis_result["analysis"])
                    
                    # Show if default results were used
                    if analysis_result.get("fallback"):
                        console.print("[yellow]⚠️ Using default analysis results (GLM response format error)[/yellow]")
                    
                    # Display understood requirements
                    console.print("\n[bold green]✅ AI understood requirements:[/bold green]")
                    req_table = Table()
                    req_table.add_column("Parameter", style="cyan")
                    req_table.add_column("Value", style="green")
                    
                    req_table.add_row("Sequence Type", requirements.get("sequence_type", "Unspecified"))
                    req_table.add_row("Target Length", f"{requirements.get('target_length', 100)}bp")
                    req_table.add_row("Functional Requirements", ", ".join(requirements.get("function_requirements", [])))
                    req_table.add_row("Special Requirements", ", ".join(requirements.get("special_requirements", [])))
                    req_table.add_row("Application Scenario", requirements.get("application_scenario", "Unspecified"))
                    
                    console.print(req_table)
                    
                    # Confirm whether to start design
                    if Confirm.ask("\nStart sequence design based on the above understanding?"):
                        # Use extracted parameters for sequence design
                        prompt = requirements.get("suggested_prompt", "TAATACGACTCACTATAGGG")
                        target_length = requirements.get("target_length", 100)
                        
                        console.print(f"\n[bold blue]🚀 Starting sequence design...[/bold blue]")
                        console.print(f"Initial prompt: {prompt}")
                        console.print(f"Target length: {target_length}bp")
                        
                        # Use new LLM+EVO2 collaborative designer
                        console.print(f"\n[bold cyan]🤖 Launching LLM+EVO2 collaborative design...[/bold cyan]")
                        
                        analyzer = SequenceAnalyzer()
                        collaborative_designer = LLMEvo2CollaborativeDesigner(
                            evo2_client=evo2_client,
                            glm_client=glm_client,
                            sequence_analyzer=analyzer
                        )
                        
                        parameters = DesignParameters(
                            initial_prompt=prompt,
                            target_length=target_length
                        )
                        
                        # Execute collaborative design
                        collaborative_result = collaborative_designer.design_sequence(
                            user_requirement=user_input,
                            design_params=parameters
                        )
                        
                        # Display design results
                        if collaborative_result.success:
                            console.print(f"\n[bold green]✅ Collaborative design completed![/bold green]")
                            console.print(f"Final sequence: {collaborative_result.final_sequence}")
                            console.print(f"Sequence length: {len(collaborative_result.final_sequence)}bp")
                            console.print(f"Quality improvement: +{collaborative_result.quality_improvement:.1f} points")
                            console.print(f"Iterations: {len(collaborative_result.iterations)} rounds")
                            
                            # Display simplified design report
                            if collaborative_result.design_report:
                                console.print("\n[bold blue]📋 Design Summary:[/bold blue]")
                                # Only display first few lines of report
                                report_lines = collaborative_result.design_report.split('\n')[:10]
                                console.print('\n'.join(report_lines))
                                if len(collaborative_result.design_report.split('\n')) > 10:
                                    console.print("[dim]... (Complete report saved)[/dim]")
                            
                            # Create project object to save results
                            from .models.project import Project, StageResult
                            import uuid
                            
                            project = Project(
                                project_id=str(uuid.uuid4())[:8],
                                parameters=parameters
                            )
                            
                            # Add collaborative design results
                            final_stage = StageResult(
                                stage=3,
                                stage_name="LLM+EVO2 Collaborative Design",
                                success=True,
                                sequence=collaborative_result.final_sequence,
                                quality_score=collaborative_result.final_analysis.get('quality_score', 0),
                                iteration=len(collaborative_result.iterations),
                                timestamp=time.time(),
                                notes=f"Natural language collaborative design, quality improvement: +{collaborative_result.quality_improvement:.1f} points"
                            )
                            
                            project.stage_results.append(final_stage)
                            project.final_sequence = collaborative_result.final_sequence
                            project.status = "completed"
                            project.completed_at = time.time()
                            
                            # Save project
                            save_project(project)
                            console.print(f"\n[green]💾 Project saved: {project.project_id}[/green]")
                        else:
                            console.print(f"[red]❌ Collaborative design failed: {collaborative_result.design_report}[/red]")
                            console.print("[yellow]🔄 Falling back to traditional design method...[/yellow]")
                            
                            # Fall back to traditional three-stage design
                            designer = ThreeStageDesigner(evo2_client, glm_client, enable_agent_optimization=False)
                            project = designer.run_complete_design(parameters)
                            
                            if project.final_sequence:
                                console.print(f"\n[bold green]✅ Traditional design completed![/bold green]")
                                console.print(f"Final sequence: {project.final_sequence}")
                                console.print(f"Sequence length: {len(project.final_sequence)}bp")
                                save_project(project)
                                console.print(f"\n[green]💾 Project saved: {project.project_id}[/green]")
                            else:
                                console.print("[red]❌ Sequence design failed[/red]")
                    
                except json.JSONDecodeError:
                    console.print("[yellow]⚠️ AI response format error, please re-describe requirements[/yellow]")
                except Exception as e:
                    import traceback
                    import logging
                    
                    # Get detailed error information
                    error_type = type(e).__name__
                    error_message = str(e)
                    error_traceback = traceback.format_exc()
                    
                    # Display user-friendly error information
                    console.print(f"[red]WARN Design process may have issues: {error_message}[/red]")
                    
                    # Display detailed technical information for debugging
                    console.print(f"[yellow]Error type: {error_type}[/yellow]")
                    console.print(f"[yellow]Detailed stack trace:[/yellow]")
                    # Ensure complete output of error stack trace
                    for line in error_traceback.split('\n'):
                        if line.strip():
                            console.print(f"[dim]{line}[/dim]")
                    
                    # Log to logging system
                    try:
                        from ..utils.logger import get_logger
                        logger = get_logger()
                        logger.log_error(
                            error_type=error_type,
                            error_message=error_message,
                            context={
                                "function": "natural_language_design",
                                "user_input": user_input,
                                "traceback": error_traceback
                            }
                        )
                    except Exception as log_error:
                        console.print(f"[dim]Log recording failed: {log_error}[/dim]")
                    
                    # Log to log file
                    logger = logging.getLogger('SequenceDesign')
                    logger.error(f"Design process exception - Type: {error_type}, Message: {error_message}")
                    logger.error(f"Full stack trace:\n{error_traceback}")
                    
                    # Add debugging tips
                    console.print("[yellow]💡 Debug Tip: Please check the above error information, or view log files for more details[/yellow]")
            else:
                console.print("[red]❌ Requirements analysis failed, please re-describe[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Exiting natural language design[/yellow]")
            break
        except Exception as e:
            import traceback
            import logging
            
            # Get detailed error information
            error_type = type(e).__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()
            
            # Display user-friendly error information
            console.print(f"[red]❌ Operation failed: {error_message}[/red]")
            
            # Display detailed technical information for debugging
            console.print(f"[dim]Error type: {error_type}[/dim]")
            console.print(f"[dim]Detailed stack trace:[/dim]")
            console.print(f"[dim]{error_traceback}[/dim]")
            
            # Log to log file
            logger = logging.getLogger('SequenceDesign')
            logger.error(f"Operation exception - Type: {error_type}, Message: {error_message}")
            logger.error(f"Full stack trace:\n{error_traceback}")
            
            # Add debugging tips and user guidance
            console.print("[yellow]💡 Debug Tip: Please check the above error information, or view log files for more details[/yellow]")
            console.print("[dim]Please re-enter or type 'exit' to end the conversation[/dim]")
    
    # Auto-save complete logs
    try:
        logger.save_json_logs()
        console.print("[dim]📄 Complete logs auto-saved[/dim]")
    except Exception as log_error:
        console.print(f"[dim]⚠️ Log save failed: {log_error}[/dim]")


# Agent configuration management functionality defined below


def interactive_analyze():
    """Interactive sequence analysis"""
    console.print("\n[bold blue]🔍 Sequence Analysis[/bold blue]")
    
    try:
        sequence = Prompt.ask("Please enter the DNA sequence to analyze")
        
        if sequence:
            analyze(sequence)
        else:
            console.print("[red]❌ Sequence cannot be empty[/red]")
    except Exception as e:
        import traceback
        import logging
        
        # Get detailed error information
        error_type = type(e).__name__
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        # Display user-friendly error information
        console.print(f"[red]❌ Sequence analysis failed: {error_message}[/red]")
        
        # Display detailed technical information for debugging
        console.print(f"[dim]Error type: {error_type}[/dim]")
        console.print(f"[dim]Detailed stack trace:[/dim]")
        console.print(f"[dim]{error_traceback}[/dim]")
        
        # Log to log file
        logger = logging.getLogger('SequenceDesign')
        logger.error(f"Sequence analysis exception - Type: {error_type}, Message: {error_message}")
        logger.error(f"Full stack trace:\n{error_traceback}")
        
        # Add debugging tips
        console.print("[yellow]💡 Debug Tip: Please check the above error information, or view log files for more details[/yellow]")
    
    # Auto-save complete logs
    try:
        logger.save_json_logs()
        console.print("[dim]📄 Complete logs auto-saved[/dim]")
    except Exception as log_error:
        console.print(f"[dim]⚠️ Log save failed: {log_error}[/dim]")


# Project management functionality removed, auto-save complete logs


# Agent configuration management functionality
def manage_agent_config():
    """Manage Agent configuration"""
    config = load_config()
    
    console.print("\n[bold blue]🤖 Agent Intelligent Optimization Configuration[/bold blue]")
    
    # Display current configuration
    current_config = config.get('agent_config', {
        'enable_optimization': True,
        'quality_threshold': 8.0,
        'max_iterations': 3,
        'improvement_threshold': 0.5,
        'auto_retry': True,
        'learning_rate': 0.1
    })
    
    config_table = Table(title="Current Agent Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_column("Description", style="dim")
    
    config_table.add_row("Enable Optimization", str(current_config['enable_optimization']), "Whether to enable Agent intelligent optimization")
    config_table.add_row("Quality Threshold", f"{current_config['quality_threshold']:.1f}", "Sequence quality score threshold (0-10)")
    config_table.add_row("Max Iterations", str(current_config['max_iterations']), "Maximum optimization iterations (1-10)")
    config_table.add_row("Improvement Threshold", f"{current_config['improvement_threshold']:.1f}", "Quality improvement threshold (0-5)")
    config_table.add_row("Auto Retry", str(current_config['auto_retry']), "Whether to auto-retry on failure")
    config_table.add_row("Learning Rate", f"{current_config['learning_rate']:.1f}", "Parameter adjustment learning rate (0-1)")
    
    console.print(config_table)
    
    if Confirm.ask("\nModify Agent configuration?"):
        try:
            # Interactive configuration update
            enable_optimization = Confirm.ask("Enable Agent intelligent optimization?", default=current_config['enable_optimization'])
            
            if enable_optimization:
                quality_threshold = float(Prompt.ask("Quality threshold (0-10)", default=str(current_config['quality_threshold'])))
                max_iterations = IntPrompt.ask("Max iterations (1-10)", default=current_config['max_iterations'])
                improvement_threshold = float(Prompt.ask("Improvement threshold (0-5)", default=str(current_config['improvement_threshold'])))
                auto_retry = Confirm.ask("Enable auto retry?", default=current_config['auto_retry'])
                learning_rate = float(Prompt.ask("Learning rate (0-1)", default=str(current_config['learning_rate'])))
                
                # Validate configuration parameters
                if not (0 < quality_threshold <= 10):
                    raise ValueError("Quality threshold must be between 0-10")
                if not (1 <= max_iterations <= 10):
                    raise ValueError("Max iterations must be between 1-10")
                if not (0 < improvement_threshold <= 5):
                    raise ValueError("Improvement threshold must be between 0-5")
                if not (0 < learning_rate <= 1):
                    raise ValueError("Learning rate must be between 0-1")
                
                agent_config = {
                    'enable_optimization': enable_optimization,
                    'quality_threshold': quality_threshold,
                    'max_iterations': max_iterations,
                    'improvement_threshold': improvement_threshold,
                    'auto_retry': auto_retry,
                    'learning_rate': learning_rate
                }
            else:
                agent_config = {'enable_optimization': False}
            
            # Save configuration
            config['agent_config'] = agent_config
            save_config(config)
            
            console.print("\n[green]✅ Agent configuration updated successfully![/green]")
            
        except Exception as e:
            import traceback
            import logging
            
            # Get detailed error information
            error_type = type(e).__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()
            
            # Display user-friendly error information
            console.print(f"\n[red]❌ Configuration update failed: {error_message}[/red]")
            
            # Display detailed technical information for debugging
            console.print(f"[dim]Error type: {error_type}[/dim]")
            console.print(f"[dim]Detailed stack trace:[/dim]")
            console.print(f"[dim]{error_traceback}[/dim]")
            
            # Log to log file
            logger = logging.getLogger('SequenceDesign')
            logger.error(f"Agent configuration update exception - Type: {error_type}, Message: {error_message}")
            logger.error(f"Full stack trace:\n{error_traceback}")
            
            # Add debugging tips
            console.print("[yellow]💡 Debug Tip: Please check the above error information, or view log files for more details[/yellow]")
    
    # Auto-save complete logs
    try:
        logger.save_json_logs()
        console.print("[dim]📄 Complete logs auto-saved[/dim]")
    except Exception as log_error:
        console.print(f"[dim]⚠️ Log save failed: {log_error}[/dim]")

@app.command()
def agent_config():
    """🤖 Configure Agent intelligent optimization parameters"""
    manage_agent_config()

if __name__ == "__main__":
    app()