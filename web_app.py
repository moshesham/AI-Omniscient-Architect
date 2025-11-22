"""Streamlit web interface for the Omniscient Architect."""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import streamlit as st
from github import GithubException

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omniscient_architect.github_client import GitHubClient, create_repository_info_from_github
from omniscient_architect.agents import GitHubRepositoryAgent
from omniscient_architect.analysis import AnalysisEngine
from omniscient_architect.models import RepositoryInfo, FileAnalysis, AgentFindings, AnalysisConfig
from omniscient_architect.logging_config import setup_logging

# Configure logging
setup_logging()

# Page configuration
st.set_page_config(
    page_title="Omniscient Architect",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .repo-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .analysis-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    """Streamlit web application for repository analysis."""

    def __init__(self):
        # Create analysis config with default values that can be overridden
        self.config = AnalysisConfig()
        self.github_client = GitHubClient()
        self.analysis_engine = AnalysisEngine(self.config)
        self.github_token = None

    def run(self):
        """Run the Streamlit application."""
        self._render_header()
        self._render_sidebar()
        self._render_main_content()

    def _render_header(self):
        """Render the application header."""
        st.markdown('<div class="main-header">🏗️ Omniscient Architect</div>', unsafe_allow_html=True)
        st.markdown("""
        **AI-Powered Repository Analysis & Code Review**

        Analyze GitHub repositories with advanced AI agents to gain deep insights into code quality,
        architecture, and development practices.
        """)

    def _render_sidebar(self):
        """Render the sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ Configuration")

            # GitHub Token Input
            github_token = st.text_input(
                "GitHub Token (Optional)",
                type="password",
                help="Provide a GitHub token for higher rate limits and private repository access"
            )

            if github_token:
                self.github_client = GitHubClient(token=github_token)
                self.github_token = github_token
                st.success("✅ GitHub token configured")

            # Analysis Configuration
            st.subheader("Analysis Settings")

            self.max_file_size = st.slider(
                "Max File Size (MB)",
                min_value=1,
                max_value=50,
                value=10,
                help="Maximum size of files to analyze"
            )

            self.max_files = st.slider(
                "Max Files to Analyze",
                min_value=10,
                max_value=1000,
                value=100,
                help="Maximum number of files to process"
            )

            self.analysis_depth = st.selectbox(
                "Analysis Depth",
                options=["quick", "standard", "deep"],
                index=1,
                help="Depth of analysis to perform"
            )

            # Update config with user selections
            self.config.max_file_size = self.max_file_size * 1024 * 1024  # Convert to bytes
            self.config.max_files = self.max_files
            self.config.analysis_depth = self.analysis_depth

            # Model Configuration
            st.subheader("AI Model")
            self.ollama_model = st.selectbox(
                "Ollama Model",
                options=["codellama:7b-instruct", "codellama:13b-instruct", "llama2:7b", "llama2:13b"],
                index=0,
                help="Select the Ollama model for analysis"
            )

            # Update config with model selection
            self.config.ollama_model = self.ollama_model

    def _render_main_content(self):
        """Render the main content area."""
        # Repository Input Section
        st.header("🔍 Repository Analysis")

        col1, col2 = st.columns([3, 1])

        with col1:
            repo_url = st.text_input(
                "GitHub Repository URL",
                placeholder="https://github.com/owner/repository",
                help="Enter the full GitHub repository URL"
            )

        with col2:
            analyze_button = st.button("🚀 Analyze Repository", type="primary", use_container_width=True)

        # Project Objective Input
        project_objective = st.text_area(
            "Project Objective (Optional)",
            height=100,
            placeholder="Describe what this project aims to accomplish...",
            help="Providing context helps the AI agents give more relevant analysis"
        )

        # Analysis Results
        if analyze_button and repo_url:
            asyncio.run(self._analyze_repository(repo_url, project_objective))
        elif analyze_button and not repo_url:
            st.error("Please enter a valid GitHub repository URL")

    async def _analyze_repository(self, repo_url: str, project_objective: str):
        """Analyze a GitHub repository."""
        try:
            with st.spinner("🔄 Connecting to repository..."):
                # Create repository info using the standalone function
                repo_info = await create_repository_info_from_github(
                    repo_url, self.github_token, project_objective or "General software development project"
                )

            st.success(f"✅ Connected to {repo_info.url}")

            # Display repository information
            self._display_repository_info(repo_info)

            # Perform analysis
            with st.spinner("🧠 AI agents analyzing repository..."):
                analysis_results = await self._run_analysis(repo_info)

            # Display results
            self._display_analysis_results(analysis_results)

        except GithubException as e:
            st.error(f"GitHub API Error: {str(e)}")
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            st.exception(e)

    def _display_repository_info(self, repo_info: RepositoryInfo):
        """Display repository information."""
        st.subheader("📊 Repository Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            repo_name = repo_info.url.split('/')[-1] if repo_info.url else "Unknown Repository"
            st.metric("Repository", repo_name)
        with col2:
            st.metric("Branch", repo_info.branch)
        with col3:
            st.metric("Type", "Remote" if repo_info.is_remote else "Local")

        if repo_info.project_objective:
            st.info(f"🎯 **Objective:** {repo_info.project_objective}")

    async def _run_analysis(self, repo_info: RepositoryInfo) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        # Initialize LLM (uses config.ollama_model)
        await self.analysis_engine.initialize_llm()

        # Run analysis with all agents
        all_findings = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create GitHub repository agent
        if self.analysis_engine.llm is None:
            raise ValueError("LLM not initialized")

        agents = [
            GitHubRepositoryAgent(self.analysis_engine.llm),
            # Add other agents as they become available
        ]

        # For now, simulate file analysis (in Phase 2, this would clone and analyze actual files)
        mock_files = self._create_mock_file_analysis(repo_info)

        # Run analysis with all agents
        all_findings = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, agent in enumerate(agents):
            status_text.text(f"🤖 Running {agent.name}...")
            findings = await agent.analyze(mock_files, repo_info)
            all_findings.append(findings)
            progress_bar.progress((i + 1) / len(agents))

        progress_bar.empty()
        status_text.empty()

        return {
            "findings": all_findings,
            "file_count": len(mock_files),
            "agents_used": len(agents)
        }

    def _create_mock_file_analysis(self, repo_info: RepositoryInfo) -> List[FileAnalysis]:
        """Create mock file analysis for demonstration (Phase 2 will analyze real files)."""
        # This is a placeholder - Phase 2 will implement actual file analysis
        mock_files = [
            FileAnalysis(
                path="README.md",
                size=2048,
                language="Markdown",
                complexity_score=1,
                issues=[],
                strengths=["Good documentation", "Clear project description"],
                content="# Project README\n\nThis is a sample project."
            ),
            FileAnalysis(
                path="src/main.py",
                size=5120,
                language="Python",
                complexity_score=3,
                issues=["Missing docstrings"],
                strengths=["Clean code structure", "Good naming conventions"],
                content="def main():\n    print('Hello World')"
            ),
            FileAnalysis(
                path="tests/test_main.py",
                size=1024,
                language="Python",
                complexity_score=2,
                issues=[],
                strengths=["Good test coverage", "Clear test cases"],
                content="def test_main():\n    assert True"
            )
        ]
        return mock_files

    def _display_analysis_results(self, results: Dict[str, Any]):
        """Display analysis results."""
        st.header("📈 Analysis Results")

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Files Analyzed", results["file_count"])
        with col2:
            st.metric("AI Agents Used", results["agents_used"])
        with col3:
            st.metric("Findings Generated", len(results["findings"]))

        # Display findings from each agent
        for finding in results["findings"]:
            self._display_agent_findings(finding)

    def _display_agent_findings(self, findings: AgentFindings):
        """Display findings from a specific agent."""
        with st.expander(f"🤖 {findings.agent_name}", expanded=True):
            st.markdown(f"**Confidence:** {findings.confidence:.2%}")
            st.markdown(f"**Reasoning:** {findings.reasoning}")

            if findings.findings:
                st.markdown("**Key Findings:**")
                for finding in findings.findings:
                    st.markdown(f"• {finding}")
            else:
                st.info("No specific findings generated")

def main():
    """Main application entry point."""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()