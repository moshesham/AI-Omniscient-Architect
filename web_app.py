"""
Omniscient Architect - AI-Powered Code Analysis Platform

A professional Streamlit interface for analyzing codebases using local LLM models.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import streamlit as st

# Add package paths for development
packages_dir = Path(__file__).parent / "packages"
for pkg in ["core", "agents", "tools", "github", "api", "llm"]:
    src_path = packages_dir / pkg / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

# Import from modular packages
from omniscient_core import FileAnalysis, RepositoryInfo
from omniscient_llm import OllamaProvider, LLMClient, ModelManager
from omniscient_agents.llm_agent import CodeReviewAgent

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Omniscient Architect",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/moshesham/AI-Omniscient-Architect',
        'Report a bug': 'https://github.com/moshesham/AI-Omniscient-Architect/issues',
        'About': '''
        # Omniscient Architect
        
        AI-Powered Code Analysis Platform with Local LLM Support.
        
        **Privacy-First**: All analysis runs locally - your code never leaves your machine.
        '''
    }
)

# =============================================================================
# Custom Styling
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-online {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-offline {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Issue cards */
    .issue-card {
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        margin: 0.75rem 0;
        border-left: 4px solid;
    }
    
    .issue-high {
        background-color: #fef2f2;
        border-color: #ef4444;
    }
    
    .issue-medium {
        background-color: #fffbeb;
        border-color: #f59e0b;
    }
    
    .issue-low {
        background-color: #f0fdf4;
        border-color: #22c55e;
    }
    
    .issue-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .issue-meta {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* File list */
    .file-item {
        padding: 0.5rem 0.75rem;
        background: #f8f9fa;
        border-radius: 0.375rem;
        margin: 0.25rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'llm_status': None,
        'available_models': [],
        'selected_model': 'qwen2.5-coder:1.5b',
        'analysis_results': None,
        'analyzed_files': [],
        'analysis_time': None,
        'focus_areas': ['security', 'architecture', 'code quality', 'best practices'],
        'max_files': 15,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Async Helpers
# =============================================================================

async def check_ollama_status() -> tuple[bool, list]:
    """Check Ollama availability and get models."""
    try:
        async with ModelManager() as manager:
            if await manager.is_available():
                models = await manager.list_models()
                return True, models
            return False, []
    except Exception:
        return False, []


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-title">🏗️ Omniscient Architect</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI-Powered Code Analysis | Privacy-First | Local LLM</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the sidebar with configuration."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # LLM Status Section
        st.markdown("### 🤖 LLM Status")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔄 Check Status", use_container_width=True):
                with st.spinner("Connecting..."):
                    is_available, models = asyncio.run(check_ollama_status())
                    st.session_state['llm_status'] = is_available
                    st.session_state['available_models'] = [m.name for m in models]
        
        # Display status
        if st.session_state['llm_status'] is True:
            st.markdown(
                '<span class="status-badge status-online">● Online</span>',
                unsafe_allow_html=True
            )
            
            if st.session_state['available_models']:
                st.session_state['selected_model'] = st.selectbox(
                    "Model",
                    options=st.session_state['available_models'],
                    index=0,
                    help="Select the Ollama model for analysis"
                )
                st.caption(f"📊 {len(st.session_state['available_models'])} model(s) available")
                
        elif st.session_state['llm_status'] is False:
            st.markdown(
                '<span class="status-badge status-offline">● Offline</span>',
                unsafe_allow_html=True
            )
            st.error("Ollama not running")
            st.code("ollama serve", language="bash")
        else:
            st.info("Click 'Check Status' to connect")
        
        st.markdown("---")
        
        # Analysis Settings
        st.markdown("### 📊 Analysis Settings")
        
        st.session_state['max_files'] = st.slider(
            "Max Files",
            min_value=5,
            max_value=50,
            value=st.session_state['max_files'],
            help="Maximum number of files to analyze"
        )
        
        st.markdown("### 🎯 Focus Areas")
        
        all_focus_areas = [
            'security', 'architecture', 'code quality', 'best practices',
            'performance', 'error handling', 'documentation', 'testing'
        ]
        
        st.session_state['focus_areas'] = st.multiselect(
            "Select focus areas",
            options=all_focus_areas,
            default=st.session_state['focus_areas'],
            help="Choose what aspects to analyze"
        )
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Actions")
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state['analysis_results'] = None
            st.session_state['analyzed_files'] = []
            st.session_state['analysis_time'] = None
            st.rerun()
        
        # Info
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption(
            "Omniscient Architect v2.0\n\n"
            "All analysis runs locally.\n"
            "Your code never leaves your machine."
        )


def find_code_files(directory: Path, max_files: int = 15) -> List[Path]:
    """Find code files in a directory."""
    code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.rb', '.cpp', '.c', '.h', '.cs'}
    exclude_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', 'dist', 'build', '.next', 'target'}
    
    files = []
    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in code_extensions:
                if not any(excluded in file_path.parts for excluded in exclude_dirs):
                    files.append(file_path)
                    if len(files) >= max_files:
                        break
    except PermissionError:
        pass
    
    return sorted(files, key=lambda x: x.suffix)


def render_local_analysis():
    """Render local directory analysis section."""
    st.markdown("### 📁 Local Directory Analysis")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        local_path = st.text_input(
            "Directory Path",
            placeholder="C:\\path\\to\\your\\project  or  /home/user/project",
            help="Enter the full path to your local repository",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_btn and local_path:
        path = Path(local_path)
        
        if not path.exists():
            st.error(f"❌ Directory not found: `{local_path}`")
            return
        
        if st.session_state['llm_status'] is not True:
            st.warning("⚠️ Please check Ollama status in the sidebar first")
            return
        
        asyncio.run(run_local_analysis(path))


async def run_local_analysis(repo_path: Path):
    """Run analysis on a local repository."""
    start_time = datetime.now()
    max_files = st.session_state.get('max_files', 15)
    model_name = st.session_state.get('selected_model', 'qwen2.5-coder:1.5b')
    focus_areas = st.session_state.get('focus_areas', ['code quality'])
    
    # Status container
    status_container = st.container()
    
    with status_container:
        # Step 1: Scan files
        with st.status("🔍 Scanning repository...", expanded=True) as status:
            code_files = find_code_files(repo_path, max_files)
            
            if not code_files:
                st.error("No code files found in the directory")
                return
            
            st.write(f"Found **{len(code_files)}** code files")
            status.update(label="📂 Loading files...", state="running")
            
            # Step 2: Load file contents
            file_analyses = []
            progress_bar = st.progress(0)
            
            ext_to_lang = {
                '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                '.jsx': 'React JSX', '.tsx': 'React TSX', '.java': 'Java',
                '.go': 'Go', '.rs': 'Rust', '.rb': 'Ruby', '.cpp': 'C++',
                '.c': 'C', '.h': 'C/C++ Header', '.cs': 'C#'
            }
            
            for i, file_path in enumerate(code_files):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    rel_path = file_path.relative_to(repo_path)
                    lang = ext_to_lang.get(file_path.suffix, 'Unknown')
                    
                    fa = FileAnalysis(
                        path=str(rel_path),
                        content=content,
                        language=lang,
                        size=len(content)
                    )
                    file_analyses.append(fa)
                except Exception:
                    pass
                
                progress_bar.progress((i + 1) / len(code_files))
            
            progress_bar.empty()
            st.session_state['analyzed_files'] = [fa.path for fa in file_analyses]
            
            # Step 3: Run LLM analysis
            status.update(label=f"🤖 Analyzing with {model_name}...", state="running")
            st.write(f"Focus: {', '.join(focus_areas)}")
            
            repo_info = RepositoryInfo(
                path=str(repo_path),
                url=f"local://{repo_path.name}",
                branch="main",
                name=repo_path.name,
                project_objective=f"Analyze code for: {', '.join(focus_areas)}"
            )
            
            try:
                provider = OllamaProvider(model=model_name)
                llm_client = LLMClient(provider=provider)
                
                async with llm_client:
                    agent = CodeReviewAgent(
                        llm_client=llm_client,
                        focus_areas=focus_areas
                    )
                    result = await agent.analyze(file_analyses, repo_info)
                    st.session_state['analysis_results'] = result
                    st.session_state['analysis_time'] = (datetime.now() - start_time).total_seconds()
                    
                status.update(label="✅ Analysis complete!", state="complete")
                
            except Exception as e:
                status.update(label="❌ Analysis failed", state="error")
                st.error(f"Error: {str(e)}")


def render_github_analysis():
    """Render GitHub repository analysis section."""
    st.markdown("### 🐙 GitHub Repository Analysis")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        github_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/owner/repository",
            help="Enter the full GitHub repository URL",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True, key="github_btn")
    
    # Optional token
    with st.expander("🔑 Authentication (Optional)"):
        github_token = st.text_input(
            "GitHub Token",
            type="password",
            help="For private repos or higher rate limits"
        )
    
    if analyze_btn and github_url:
        if st.session_state['llm_status'] is not True:
            st.warning("⚠️ Please check Ollama status in the sidebar first")
            return
        
        asyncio.run(run_github_analysis(github_url, github_token if 'github_token' in dir() else None))


async def run_github_analysis(repo_url: str, token: Optional[str] = None):
    """Run analysis on a GitHub repository."""
    try:
        from omniscient_github import GitHubClient, parse_github_url
        
        start_time = datetime.now()
        owner, repo_name = parse_github_url(repo_url)
        max_files = st.session_state.get('max_files', 15)
        model_name = st.session_state.get('selected_model', 'qwen2.5-coder:1.5b')
        focus_areas = st.session_state.get('focus_areas', ['code quality'])
        
        with st.status("🐙 Connecting to GitHub...", expanded=True) as status:
            async with GitHubClient(token=token) as client:
                repo_data = await client.get_repository(owner, repo_name)
                st.write(f"**{repo_data.name}** | {repo_data.language or 'Mixed'}")
                
                status.update(label="📂 Fetching files...", state="running")
                files = await client.list_files_recursive(owner, repo_name)
                
                code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.rb'}
                code_files = [f for f in files if f.type == 'file' and 
                             any(f.path.endswith(ext) for ext in code_extensions)][:max_files]
                
                if not code_files:
                    st.warning("No code files found")
                    return
                
                st.write(f"Found **{len(code_files)}** code files")
                
                # Fetch contents
                status.update(label="📥 Downloading files...", state="running")
                progress = st.progress(0)
                file_analyses = []
                
                lang_map = {
                    '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                    '.jsx': 'React JSX', '.tsx': 'React TSX', '.java': 'Java',
                    '.go': 'Go', '.rs': 'Rust', '.rb': 'Ruby'
                }
                
                for i, file_info in enumerate(code_files):
                    try:
                        content = await client.get_file_content(owner, repo_name, file_info.path)
                        if content:
                            ext = '.' + file_info.path.split('.')[-1] if '.' in file_info.path else ''
                            fa = FileAnalysis(
                                path=file_info.path,
                                content=content,
                                language=lang_map.get(ext, 'Unknown'),
                                size=len(content)
                            )
                            file_analyses.append(fa)
                    except Exception:
                        pass
                    progress.progress((i + 1) / len(code_files))
                
                progress.empty()
                st.session_state['analyzed_files'] = [fa.path for fa in file_analyses]
                
                # Run analysis
                status.update(label=f"🤖 Analyzing with {model_name}...", state="running")
                
                repo_info = RepositoryInfo(
                    path=repo_url,
                    url=repo_url,
                    branch=repo_data.default_branch,
                    name=repo_name,
                    project_objective=f"Analyze code for: {', '.join(focus_areas)}"
                )
                
                provider = OllamaProvider(model=model_name)
                llm_client = LLMClient(provider=provider)
                
                async with llm_client:
                    agent = CodeReviewAgent(llm_client=llm_client, focus_areas=focus_areas)
                    result = await agent.analyze(file_analyses, repo_info)
                    st.session_state['analysis_results'] = result
                    st.session_state['analysis_time'] = (datetime.now() - start_time).total_seconds()
                
                status.update(label="✅ Analysis complete!", state="complete")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def render_results():
    """Render analysis results."""
    results = st.session_state.get('analysis_results')
    if not results:
        # Empty state
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem; color: #666;">
                <h3>👋 Ready to Analyze</h3>
                <p>Enter a local directory path or GitHub URL above to get started.</p>
                <p style="font-size: 0.9rem;">Make sure Ollama is running and check the status in the sidebar.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        files_count = len(st.session_state.get('analyzed_files', []))
        st.metric("Files Analyzed", files_count)
    
    with col2:
        issues_count = len(results.issues) if hasattr(results, 'issues') and results.issues else 0
        st.metric("Issues Found", issues_count)
    
    with col3:
        analysis_time = st.session_state.get('analysis_time', 0)
        st.metric("Analysis Time", f"{analysis_time:.1f}s")
    
    # Summary
    if hasattr(results, 'summary') and results.summary:
        st.markdown("### 📋 Summary")
        st.info(results.summary)
    
    # Issues
    if hasattr(results, 'issues') and results.issues:
        st.markdown("### 🔍 Issues")
        
        for issue in results.issues:
            # Handle both dict and object formats
            if isinstance(issue, dict):
                severity = str(issue.get('severity', 'medium')).lower()
                category = issue.get('category', 'General')
                description = issue.get('description', '')
                file_path = issue.get('file_path', '')
                line_number = issue.get('line_number', '')
                suggestion = issue.get('suggestion', '')
            else:
                severity = str(getattr(issue, 'severity', 'medium')).lower()
                category = getattr(issue, 'category', 'General')
                description = getattr(issue, 'description', '')
                file_path = getattr(issue, 'file_path', '')
                line_number = getattr(issue, 'line_number', '')
                suggestion = getattr(issue, 'suggestion', '')
            
            severity_class = f"issue-{severity}" if severity in ['high', 'medium', 'low'] else 'issue-medium'
            severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢', 'critical': '🔴'}.get(severity, '⚪')
            
            st.markdown(
                f"""
                <div class="issue-card {severity_class}">
                    <div class="issue-title">{severity_icon} [{severity.upper()}] {category}</div>
                    <div>{description}</div>
                    <div class="issue-meta">
                        {'📁 ' + str(file_path) if file_path else ''}
                        {' | Line ' + str(line_number) if line_number else ''}
                    </div>
                    {'<div style="margin-top: 0.5rem;"><strong>💡 Suggestion:</strong> ' + str(suggestion) + '</div>' if suggestion else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Recommendations
    if hasattr(results, 'recommendations') and results.recommendations:
        st.markdown("### 💡 Recommendations")
        for rec in results.recommendations:
            st.markdown(f"- {rec}")
    
    # Files analyzed
    if st.session_state.get('analyzed_files'):
        with st.expander(f"📁 Files Analyzed ({len(st.session_state['analyzed_files'])})"):
            for f in st.session_state['analyzed_files']:
                st.markdown(f'<div class="file-item">{f}</div>', unsafe_allow_html=True)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    render_sidebar()
    
    # Main content with tabs
    tab1, tab2 = st.tabs(["📁 Local Analysis", "🐙 GitHub Analysis"])
    
    with tab1:
        render_local_analysis()
    
    with tab2:
        render_github_analysis()
    
    # Results section
    render_results()


if __name__ == "__main__":
    main()
