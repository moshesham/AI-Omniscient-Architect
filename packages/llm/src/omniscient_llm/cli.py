"""CLI for LLM model management."""

import asyncio
import sys
from typing import Optional

import click

from omniscient_llm.manager import ModelManager


def run_async(coro):
    """Run async function in sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)


@click.group()
@click.option(
    "--url",
    default="http://localhost:11434",
    help="Ollama server URL",
)
@click.pass_context
def cli(ctx, url: str):
    """Manage local LLM models via Ollama."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url


@cli.command("list")
@click.pass_context
def list_models(ctx):
    """List available local models."""
    async def _list():
        async with ModelManager(base_url=ctx.obj["url"]) as manager:
            models = await manager.list_models()
            
            if not models:
                click.echo("No models installed.")
                click.echo("\nRun 'llm pull <model>' to download a model.")
                click.echo("Suggestions: llama3.2:latest, codellama:7b, mistral:latest")
                return
            
            click.echo(f"\n{'Model':<40} {'Size':>10} {'Modified':<20}")
            click.echo("-" * 72)
            
            for model in sorted(models, key=lambda m: m.name):
                size = f"{model.size_bytes / (1024**3):.1f} GB" if model.size_bytes else "Unknown"
                modified = model.modified_at.strftime("%Y-%m-%d %H:%M") if model.modified_at else "Unknown"
                click.echo(f"{model.name:<40} {size:>10} {modified:<20}")
            
            click.echo(f"\nTotal: {len(models)} model(s)")
    
    try:
        run_async(_list())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("pull")
@click.argument("name")
@click.pass_context
def pull_model(ctx, name: str):
    """Pull a model from the registry."""
    async def _pull():
        async with ModelManager(base_url=ctx.obj["url"]) as manager:
            click.echo(f"Pulling {name}...")
            
            last_percent = -1
            last_status = ""
            
            async for progress in manager.pull_model(name):
                if progress.total > 0:
                    percent = int(progress.percent)
                    if percent != last_percent or progress.status != last_status:
                        last_percent = percent
                        last_status = progress.status
                        bar = "=" * (percent // 2) + ">" + " " * (50 - percent // 2)
                        click.echo(
                            f"\r[{bar}] {percent:3d}% - "
                            f"{progress.downloaded_mb:.1f}/{progress.size_mb:.1f} MB - "
                            f"{progress.status[:30]:<30}",
                            nl=False,
                        )
                else:
                    if progress.status != last_status:
                        last_status = progress.status
                        click.echo(f"\r{progress.status:<80}", nl=False)
            
            click.echo("\n✓ Model pulled successfully!")
    
    try:
        run_async(_pull())
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)


@cli.command("delete")
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_model(ctx, name: str, force: bool):
    """Delete a local model."""
    async def _delete():
        async with ModelManager(base_url=ctx.obj["url"]) as manager:
            # Check if model exists
            model = await manager.get_model(name)
            if not model:
                click.echo(f"Model '{name}' not found.", err=True)
                sys.exit(1)
            
            if not force:
                size = f"{model.size_bytes / (1024**3):.1f} GB" if model.size_bytes else "Unknown"
                click.echo(f"Model: {model.name}")
                click.echo(f"Size: {size}")
                if not click.confirm("Delete this model?"):
                    click.echo("Cancelled.")
                    return
            
            await manager.delete_model(name)
            click.echo(f"✓ Model '{name}' deleted.")
    
    try:
        run_async(_delete())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("info")
@click.argument("name")
@click.pass_context
def model_info(ctx, name: str):
    """Show detailed model information."""
    async def _info():
        async with ModelManager(base_url=ctx.obj["url"]) as manager:
            try:
                info = await manager.get_model_info(name)
            except Exception as e:
                click.echo(f"Model '{name}' not found.", err=True)
                sys.exit(1)
            
            click.echo(f"\n{'='*60}")
            click.echo(f"Model: {name}")
            click.echo(f"{'='*60}")
            
            if "license" in info:
                click.echo(f"\nLicense: {info['license'][:100]}...")
            
            if "modelfile" in info:
                click.echo(f"\nModelfile:\n{info['modelfile']}")
            
            if "parameters" in info:
                click.echo(f"\nParameters:\n{info['parameters']}")
            
            if "template" in info:
                click.echo(f"\nTemplate:\n{info['template']}")
    
    try:
        run_async(_info())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("recommend")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["code", "general", "small", "analysis"]),
    default="general",
    help="Model category",
)
@click.option(
    "--max-size",
    "-s",
    type=float,
    default=None,
    help="Maximum size in GB",
)
def recommend(category: str, max_size: Optional[float]):
    """Show recommended models for a use case."""
    manager = ModelManager()
    models = manager.get_recommendations(category, max_size)
    
    click.echo(f"\nRecommended models for {category}:")
    click.echo("-" * 40)
    
    for i, model in enumerate(models, 1):
        click.echo(f"  {i}. {model}")
    
    click.echo(f"\nRun 'llm pull <model>' to download.")


@cli.command("status")
@click.pass_context
def status(ctx):
    """Check Ollama server status."""
    async def _status():
        manager = ModelManager(base_url=ctx.obj["url"])
        
        try:
            await manager.initialize()
            is_available = await manager.is_available()
            
            if is_available:
                models = await manager.list_models()
                click.echo("✓ Ollama server is running")
                click.echo(f"  URL: {ctx.obj['url']}")
                click.echo(f"  Models: {len(models)} installed")
            else:
                click.echo("✗ Ollama server not responding")
                sys.exit(1)
                
        except Exception as e:
            click.echo(f"✗ Cannot connect to Ollama: {e}")
            click.echo(f"\nMake sure Ollama is running: ollama serve")
            sys.exit(1)
        finally:
            await manager.close()
    
    run_async(_status())


@cli.command("copy")
@click.argument("source")
@click.argument("destination")
@click.pass_context
def copy_model(ctx, source: str, destination: str):
    """Copy/rename a model."""
    async def _copy():
        async with ModelManager(base_url=ctx.obj["url"]) as manager:
            success = await manager.copy_model(source, destination)
            if success:
                click.echo(f"✓ Copied {source} -> {destination}")
            else:
                click.echo("Failed to copy model.", err=True)
                sys.exit(1)
    
    try:
        run_async(_copy())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
