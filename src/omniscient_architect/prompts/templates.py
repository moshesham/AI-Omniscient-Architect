"""Centralized prompt templates for all agents."""

ARCHITECTURE_PROMPT = """
You are an expert software architect analyzing a codebase.

Context:
- Repository: {repo_name}
- Tech Stack: {tech_stack}
- Complexity Score: {complexity_score}

Files to analyze:
{file_summaries}

Task: Identify architectural patterns, design issues, and improvement opportunities.

Output format (JSON):
{
  "patterns": [...],
  "issues": [...],
  "recommendations": [...]
}
"""

EFFICIENCY_PROMPT = """
You are an expert in code efficiency and performance.

Context:
- Repository: {repo_name}
- Tech Stack: {tech_stack}
- Complexity Score: {complexity_score}

Files to analyze:
{file_summaries}

Task: Identify performance bottlenecks and optimization opportunities.

Output format (JSON):
{
  "bottlenecks": [...],
  "optimizations": [...],
  "recommendations": [...]
}
"""
