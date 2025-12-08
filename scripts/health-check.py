#!/usr/bin/env python3
"""
Health Check and Monitoring Script for Omniscient Architect

This script checks the health of all services and provides detailed status information.
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, List
import httpx
import psycopg


class HealthChecker:
    """Health check coordinator for all services."""
    
    def __init__(
        self,
        postgres_url: str = "postgresql://omniscient:localdev@localhost:5432/omniscient",
        ollama_url: str = "http://localhost:11434",
        app_url: str = "http://localhost:8501",
    ):
        self.postgres_url = postgres_url
        self.ollama_url = ollama_url
        self.app_url = app_url
        self.results: Dict[str, Dict] = {}
    
    async def check_postgres(self) -> Dict:
        """Check PostgreSQL database health."""
        try:
            start = datetime.now()
            async with await psycopg.AsyncConnection.connect(self.postgres_url) as conn:
                # Check connection
                async with conn.cursor() as cur:
                    await cur.execute("SELECT version()")
                    version = await cur.fetchone()
                    
                    # Check extensions
                    await cur.execute("""
                        SELECT extname FROM pg_extension 
                        WHERE extname IN ('vector', 'pg_trgm')
                    """)
                    extensions = [row[0] for row in await cur.fetchall()]
                    
                    # Check schema
                    await cur.execute("""
                        SELECT schema_name FROM information_schema.schemata 
                        WHERE schema_name = 'rag'
                    """)
                    schema_exists = await cur.fetchone() is not None
                    
                    # Count documents and chunks
                    if schema_exists:
                        await cur.execute("SELECT COUNT(*) FROM rag.documents")
                        doc_count = (await cur.fetchone())[0]
                        
                        await cur.execute("SELECT COUNT(*) FROM rag.chunks")
                        chunk_count = (await cur.fetchone())[0]
                        
                        await cur.execute("SELECT COUNT(*) FROM rag.learned_facts")
                        fact_count = (await cur.fetchone())[0]
                    else:
                        doc_count = chunk_count = fact_count = 0
            
            duration = (datetime.now() - start).total_seconds()
            
            return {
                "status": "healthy",
                "response_time_ms": duration * 1000,
                "version": version[0] if version else "unknown",
                "extensions": extensions,
                "schema_exists": schema_exists,
                "documents": doc_count,
                "chunks": chunk_count,
                "learned_facts": fact_count,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    async def check_ollama(self) -> Dict:
        """Check Ollama service health."""
        try:
            start = datetime.now()
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check API availability
                response = await client.get(f"{self.ollama_url}/api/tags")
                response.raise_for_status()
                
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                # Check if embedding model is available
                has_embedding = any("nomic-embed-text" in m for m in model_names)
            
            duration = (datetime.now() - start).total_seconds()
            
            return {
                "status": "healthy",
                "response_time_ms": duration * 1000,
                "models": model_names,
                "has_embedding_model": has_embedding,
                "model_count": len(model_names),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    async def check_app(self) -> Dict:
        """Check Streamlit application health."""
        try:
            start = datetime.now()
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check health endpoint
                response = await client.get(f"{self.app_url}/_stcore/health")
                response.raise_for_status()
            
            duration = (datetime.now() - start).total_seconds()
            
            return {
                "status": "healthy",
                "response_time_ms": duration * 1000,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    async def run_all_checks(self) -> Dict[str, Dict]:
        """Run all health checks in parallel."""
        print("🏥 Running health checks...")
        print("-" * 60)
        
        # Run checks in parallel
        postgres_task = asyncio.create_task(self.check_postgres())
        ollama_task = asyncio.create_task(self.check_ollama())
        app_task = asyncio.create_task(self.check_app())
        
        self.results = {
            "postgres": await postgres_task,
            "ollama": await ollama_task,
            "app": await app_task,
            "timestamp": datetime.now().isoformat(),
        }
        
        return self.results
    
    def print_results(self):
        """Print health check results in a readable format."""
        print()
        print("📊 Health Check Results")
        print("=" * 60)
        
        for service, result in self.results.items():
            if service == "timestamp":
                continue
            
            status = result.get("status", "unknown")
            emoji = "✅" if status == "healthy" else "❌"
            
            print(f"\n{emoji} {service.upper()}")
            print("-" * 40)
            
            if status == "healthy":
                print(f"  Status: {status}")
                if "response_time_ms" in result:
                    print(f"  Response Time: {result['response_time_ms']:.2f} ms")
                
                # Service-specific details
                if service == "postgres":
                    print(f"  Documents: {result.get('documents', 0)}")
                    print(f"  Chunks: {result.get('chunks', 0)}")
                    print(f"  Learned Facts: {result.get('learned_facts', 0)}")
                    print(f"  Extensions: {', '.join(result.get('extensions', []))}")
                elif service == "ollama":
                    print(f"  Models: {result.get('model_count', 0)}")
                    print(f"  Embedding Model: {'✓' if result.get('has_embedding_model') else '✗'}")
                    if result.get('models'):
                        print(f"  Available: {', '.join(result['models'][:3])}")
            else:
                print(f"  Status: {status}")
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print()
        print("=" * 60)
        
        # Overall status
        all_healthy = all(
            r.get("status") == "healthy"
            for k, r in self.results.items()
            if k != "timestamp"
        )
        
        if all_healthy:
            print("✅ All services are healthy!")
            return 0
        else:
            print("❌ Some services are unhealthy!")
            return 1
    
    def export_json(self, filename: str = "health-report.json"):
        """Export results to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Results exported to {filename}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health check for Omniscient Architect")
    parser.add_argument("--postgres", default="postgresql://omniscient:localdev@localhost:5432/omniscient")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--app", default="http://localhost:8501")
    parser.add_argument("--json", help="Export results to JSON file")
    
    args = parser.parse_args()
    
    checker = HealthChecker(
        postgres_url=args.postgres,
        ollama_url=args.ollama,
        app_url=args.app,
    )
    
    await checker.run_all_checks()
    exit_code = checker.print_results()
    
    if args.json:
        checker.export_json(args.json)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
