"""Command-line interface for the Omniscient Architect."""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from .analysis import AnalysisEngine
from .models import RepositoryInfo, AnalysisConfig
from .config import load_config
from .reporting import ReportGenerator


console = Console()


class CLI:
    """Command-line interface for the Omniscient Architect."""

    def __init__(self):
        self.engine: Optional[AnalysisEngine] = None

    async def run(self, args: argparse.Namespace) -> int:
        """Run the CLI with parsed arguments."""
        try:
            # Initialize analysis engine (config from config.yaml / env, with CLI overrides)
            cli_overrides = {}
            if args.model:
                cli_overrides["ollama_model"] = args.model
            if args.depth:
                cli_overrides["analysis_depth"] = args.depth

            config = load_config(cli_overrides)
            self.engine = AnalysisEngine(config)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                init_task = progress.add_task("Initializing AI engine...", total=None)

                if not await self.engine.initialize_llm():
                    console.print("[red]Failed to initialize LLM. Please ensure Ollama is running.[/red]")
                    return 1

                progress.update(init_task, completed=True)

            # Validate repository path
            repo_path = Path(args.repo_path)
            if not repo_path.exists():
                console.print(f"[red]Error: Repository path does not exist: {repo_path}[/red]")
                return 1

            # Create repository info
            repo_info = RepositoryInfo(
                path=repo_path,
                project_objective=args.objective or ""
            )

            # Run analysis
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                analysis_task = progress.add_task("Analyzing repository...", total=None)

                result = await self.engine.analyze_repository(repo_info)

                progress.update(analysis_task, completed=True)

            # Generate and output report
            report_generator = ReportGenerator()
            report = report_generator.generate_markdown_report(result)

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(report, encoding='utf-8')
                console.print(f"[green]Report saved to: {output_path}[/green]")
            else:
                console.print(Panel.fit(
                    Text(report, style="dim"),
                    title="🧠 Omniscient Architect Report",
                    border_style="blue"
                ))

            return 0

        except KeyboardInterrupt:
            console.print("[yellow]Analysis interrupted by user.[/yellow]")
            return 130
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="omniscient-architect",
            description="AI-powered code review and development assistant",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Analyze current directory
  omniscient-architect .

  # Analyze with specific objective
  omniscient-architect /path/to/repo --objective "Build a user authentication system"

  # Save output to file
  omniscient-architect . --objective "Your goal" --output review.md

  # Use different model
  omniscient-architect . --model llama2:13b

  # Quick analysis
  omniscient-architect . --depth quick
            """
        )

        parser.add_argument(
            'repo_path',
            help='Path to the repository to analyze'
        )

        parser.add_argument(
            '-o', '--objective',
            help='Project objective to analyze against',
            default=''
        )

        parser.add_argument(
            '--output',
            help='Output file path (default: print to stdout)',
            default=None
        )

        parser.add_argument(
            '--model',
            help='Ollama model to use (default: codellama:7b-instruct)',
            default=None
        )

        parser.add_argument(
            '--depth',
            choices=['quick', 'standard', 'deep'],
            help='Analysis depth (default: standard)',
            default='standard'
        )

        return parser


async def main() -> int:
    """Main entry point."""
    parser = CLI.create_parser()
    args = parser.parse_args()

    cli = CLI()
    return await cli.run(args)


def sync_main() -> None:
    """Synchronous wrapper for the main function."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)


if __name__ == '__main__':
    sync_main()