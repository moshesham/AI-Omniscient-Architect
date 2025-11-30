"""CLI for running the API server."""

import argparse
import sys


def main():
    """CLI entry point for Omniscient API."""
    parser = argparse.ArgumentParser(
        prog="omniscient-api",
        description="Omniscient Architect API Server",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run the API server")
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    serve_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers (default: 1)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.add_argument(
        "--config",
        help="Path to configuration file",
    )
    
    # Version command
    subparsers.add_parser("version", help="Show version")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        from omniscient_api.app import run_server
        run_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            config_path=args.config,
        )
    elif args.command == "version":
        from omniscient_api import __version__
        print(f"omniscient-api {__version__}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
