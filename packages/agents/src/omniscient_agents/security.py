"""Security analysis agent."""

from omniscient_core import BaseAIAgent

from .prompts import load_prompt


class SecurityAgent(BaseAIAgent):
    """Agent for security vulnerability analysis.

    Analyzes codebases for:
    - Injection vulnerabilities (SQL, command, path traversal)
    - Authentication and authorization weaknesses
    - Hardcoded secrets, tokens, and credentials
    - Insecure cryptographic practices
    - Dependency vulnerabilities and outdated packages
    - OWASP Top-10 issues
    - Sensitive data exposure
    - Insecure direct object references
    - Security misconfiguration
    - Cross-site scripting (XSS) and request forgery (CSRF)
    """

    def get_prompt_template(self) -> str:
        """Load the security prompt template."""
        return load_prompt("security")

    def get_default_objective(self) -> str:
        """Get default analysis objective."""
        return (
            "Identify security vulnerabilities, insecure patterns, and "
            "hardcoded secrets that could expose the application to attack."
        )
