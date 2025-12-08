"""
Omniscient Architect - AI-Powered Code Analysis Platform

A professional Streamlit interface for analyzing codebases using local LLM models.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import streamlit as st

# Add package paths
_r = Path(__file__).parent
for _p in ["core", "agents", "tools", "github", "api", "llm", "rag"]:
    _path = _r / "packages" / _p / "src"
    if _path.exists(): sys.path.insert(0, str(_path))

# Import from modular packages
from omniscient_core import FileAnalysis, RepositoryInfo, CODE_EXTENSIONS, EXCLUDE_DIRS
from omniscient_llm import OllamaProvider, LLMClient, ModelManager
from omniscient_agents.llm_agent import CodeReviewAgent

# Try to import RAG components
try:
    from omniscient_rag import RAGPipeline, RAGConfig, ChunkerFactory
    from omniscient_rag.store import PostgresVectorStore, DatabaseConfig
    from omniscient_rag.metrics import KnowledgeScorer
    from omniscient_rag.learning import (
        KnowledgeMemory,
        FeedbackLearner,
        ContextInjector,
        InjectionConfig,
        FeedbackType,
    )
    HAS_RAG = True
    HAS_LEARNING = True
except ImportError as e:
    HAS_RAG = False
    HAS_LEARNING = False

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
        # RAG Knowledge Base state
        'rag_initialized': False,
        'knowledge_stats': None,
        'knowledge_score': None,
        'chunking_strategy': 'auto',
        'chunk_size': 512,
        'hybrid_alpha': 0.5,
        # Learning state
        'current_query': None,
        'current_answer': None,
        'retrieved_chunks': None,
        'learned_knowledge': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Async Helpers
# =============================================================================

class ResourceManager:
    """Context manager for database and LLM resources."""
    
    def __init__(self, db_url: str, model_name: str):
        self.db_url = db_url
        self.model_name = model_name
        self.provider = None
        self.store = None
    
    async def __aenter__(self):
        """Initialize resources."""
        self.provider = OllamaProvider(model=self.model_name)
        await self.provider.initialize()
        
        self.store = PostgresVectorStore(DatabaseConfig(connection_string=self.db_url))
        await self.store.initialize()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        if self.store:
            try:
                await self.store.close()
            except Exception:
                pass
        if self.provider:
            try:
                await self.provider.close()
            except Exception:
                pass
    
    async def embed(self, text: str):
        """Embed text using the provider."""
        return await self.provider.embed(text)
    
    async def generate(self, prompt: str) -> str:
        """Generate text using the provider."""
        response = await self.provider.generate(prompt)
        return response.content


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
    files = []
    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in CODE_EXTENSIONS:
                if not any(excluded in file_path.parts for excluded in EXCLUDE_DIRS):
                    files.append(file_path)
                    if len(files) >= max_files:
                        break
    except PermissionError:
        pass  # Skip directories we cannot access
    
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
                    pass  # Skip files that cannot be read
                
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
                
                code_files = [f for f in files if f.type == 'file' and 
                             any(f.path.endswith(ext) for ext in CODE_EXTENSIONS)][:max_files]
                
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
                        pass  # Skip files that cannot be fetched
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
    tab1, tab2, tab3 = st.tabs(["📁 Local Analysis", "🐙 GitHub Analysis", "📚 Knowledge Base"])
    
    with tab1:
        render_local_analysis()
    
    with tab2:
        render_github_analysis()
    
    with tab3:
        render_knowledge_base()
    
    # Results section
    render_results()


def render_knowledge_base():
    """Render the Knowledge Base / RAG section."""
    st.markdown("### 📚 Knowledge Base")
    st.markdown("Upload domain-specific documentation to enhance AI analysis with expert knowledge.")
    
    if not HAS_RAG:
        st.warning("⚠️ RAG module not available. Install with: `pip install omniscient-rag`")
        return
    
    # Check database connection
    import os
    db_url = os.getenv("DATABASE_URL", "postgresql://omniscient:localdev@localhost:5432/omniscient")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### 📊 Knowledge Base Status")
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_kb"):
            asyncio.run(refresh_knowledge_stats(db_url))
    
    # Display stats
    stats = st.session_state.get('knowledge_stats')
    if stats:
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Documents", stats.get('documents', 0))
        with metric_cols[1]:
            st.metric("Chunks", stats.get('chunks', 0))
        with metric_cols[2]:
            st.metric("Test Questions", stats.get('knowledge_questions', 0))
        with metric_cols[3]:
            latest = stats.get('latest_score', {})
            score = latest.get('overall', 0) if latest else 0
            st.metric("Knowledge Score", f"{score:.1f}%")
    else:
        st.info("Click 'Refresh' to load knowledge base statistics.")
    
    st.markdown("---")
    
    # Ingestion Section
    st.markdown("#### 📥 Ingest Documents")
    
    # Chunking strategy selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state['chunking_strategy'] = st.selectbox(
            "Chunking Strategy",
            options=['auto', 'fixed', 'semantic', 'ast'],
            index=['auto', 'fixed', 'semantic', 'ast'].index(st.session_state['chunking_strategy']),
            help="""
            **auto**: Auto-detect based on file type (recommended)
            **fixed**: Token-based with fixed size
            **semantic**: Split by headings/paragraphs
            **ast**: Split by code structure (functions/classes)
            """
        )
    
    with col2:
        st.session_state['chunk_size'] = st.select_slider(
            "Chunk Size (tokens)",
            options=[256, 512, 768, 1024],
            value=st.session_state['chunk_size'],
            help="Target size for each chunk"
        )
    
    with col3:
        st.session_state['hybrid_alpha'] = st.slider(
            "Hybrid Alpha",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state['hybrid_alpha'],
            step=0.1,
            help="0 = BM25 only, 1 = Vector only, 0.5 = balanced"
        )
    
    # Directory input for ingestion
    ingest_path = st.text_input(
        "Document Folder",
        placeholder="C:\\path\\to\\documentation or /path/to/docs",
        help="Path to folder containing documentation (*.md, *.txt, *.py, etc.)"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📥 Ingest Documents", type="primary", use_container_width=True):
            if ingest_path:
                path = Path(ingest_path)
                if path.exists() and path.is_dir():
                    asyncio.run(ingest_documents(path, db_url))
                else:
                    st.error("❌ Invalid directory path")
            else:
                st.warning("Please enter a folder path")
    
    with col2:
        if st.button("📊 Evaluate Knowledge", use_container_width=True):
            asyncio.run(evaluate_knowledge(db_url))
    
    # View Learned Knowledge section
    if HAS_LEARNING:
        with st.expander("🧠 View Learned Knowledge"):
            st.markdown("See what the AI has learned from previous interactions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📚 View Learned Facts", use_container_width=True):
                    asyncio.run(view_learned_facts(db_url))
            with col2:
                if st.button("🔄 View Query Patterns", use_container_width=True):
                    asyncio.run(view_query_refinements(db_url))
            
            # Display learned facts
            learned_facts = st.session_state.get('learned_facts', [])
            if learned_facts:
                st.markdown("**Learned Facts:**")
                for fact in learned_facts[:10]:  # Show top 10
                    confidence = fact.get('confidence', 0)
                    st.markdown(f"- **{fact.get('topic', 'Unknown')}**: {fact.get('fact', '')}")
                    st.caption(f"Confidence: {confidence:.0%} | Used: {fact.get('usage_count', 0)} times | Success: {fact.get('success_rate', 0):.0%}")
            
            # Display query refinements
            refinements = st.session_state.get('query_refinements', [])
            if refinements:
                st.markdown("**Query Refinement Patterns:**")
                for ref in refinements[:5]:  # Show top 5
                    st.markdown(f"- '{ref.get('original_query', '')}' → '{ref.get('refined_query', '')}'")
                    st.caption(f"Applied: {ref.get('times_applied', 0)} times | Success: {ref.get('success_rate', 0):.0%}")
    
    st.markdown("---")
    
    # Query Section
    st.markdown("#### 🔍 Query Knowledge Base")
    
    query_text = st.text_input(
        "Ask a question",
        placeholder="How do I configure Spark executors for optimal performance?",
        help="Query the knowledge base with natural language"
    )
    
    if st.button("🔎 Search", use_container_width=False) and query_text:
        asyncio.run(query_knowledge_base(query_text, db_url))
    
    # Display AI answer and feedback
    current_answer = st.session_state.get('current_answer')
    current_query = st.session_state.get('current_query')
    
    if current_answer and current_query:
        st.markdown("---")
        st.markdown("##### 🤖 AI Answer")
        
        # Show learned knowledge context if available
        learned_knowledge = st.session_state.get('learned_knowledge')
        if learned_knowledge and HAS_LEARNING:
            with st.expander("🧠 Using Learned Knowledge"):
                st.info(f"The AI is using {learned_knowledge.count('Learned Fact') if learned_knowledge else 0} facts from previous interactions")
                st.markdown(learned_knowledge)
        
        st.markdown(current_answer)
        
        # Feedback section
        if HAS_LEARNING:
            st.markdown("##### 📝 Rate this answer to help the AI learn")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("👍 Helpful", key="feedback_positive", use_container_width=True):
                    asyncio.run(submit_feedback(db_url, rating=0.9, feedback_type="positive"))
                    
            with col2:
                rating = st.slider("Custom Rating", 1, 5, 3, key="custom_rating")
                if st.button("Submit Rating", key="feedback_rating", use_container_width=True):
                    asyncio.run(submit_feedback(db_url, rating=rating/5.0, feedback_type="neutral"))
                    
            with col3:
                if st.button("👎 Not Helpful", key="feedback_negative", use_container_width=True):
                    asyncio.run(submit_feedback(db_url, rating=-0.5, feedback_type="negative"))
            
            # Optional correction field
            correction = st.text_area("💡 Provide the correct answer (optional):", key="correction_text", height=100)
            if correction and st.button("Submit Correction", key="submit_correction"):
                asyncio.run(submit_feedback(db_url, rating=-1.0, feedback_type="correction", correction=correction))
        
        st.markdown("---")
    
    # Display retrieved sources
    query_results = st.session_state.get('query_results')
    if query_results:
        st.markdown("##### 📚 Retrieved Sources")
        for i, result in enumerate(query_results, 1):
            with st.expander(f"#{i} - {result.get('source', 'Unknown')} (score: {result.get('score', 0):.3f})"):
                st.markdown(result.get('content', ''))


async def refresh_knowledge_stats(db_url: str):
    """Refresh knowledge base statistics."""
    try:
        store = PostgresVectorStore(DatabaseConfig(connection_string=db_url))
        await store.initialize()
        stats = await store.get_stats()
        await store.close()
        st.session_state['knowledge_stats'] = stats
        st.session_state['rag_initialized'] = True
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.session_state['knowledge_stats'] = None


async def ingest_documents(path: Path, db_url: str):
    """Ingest documents from a folder."""
    model_name = st.session_state.get('selected_model', 'qwen2.5-coder:1.5b')
    chunking = st.session_state.get('chunking_strategy', 'auto')
    chunk_size = st.session_state.get('chunk_size', 512)
    
    with st.status("📥 Ingesting documents...", expanded=True) as status:
        try:
            # Initialize provider for embeddings
            provider = OllamaProvider(model=model_name)
            await provider.initialize()
            
            # Create pipeline
            config = RAGConfig(
                chunking_strategy=chunking,
                chunk_size=chunk_size,
                embedding_model="nomic-embed-text",
                auto_generate_questions=True,
                questions_per_document=3,
            )
            
            store = PostgresVectorStore(DatabaseConfig(connection_string=db_url))
            chunker = ChunkerFactory.create(chunking, chunk_size)
            
            pipeline = RAGPipeline(
                store=store,
                chunker=chunker,
                embed_fn=provider.embed,
                llm_fn=None,  # Skip question generation without LLM
                config=config,
            )
            
            await pipeline.initialize()
            
            # Progress callback
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_cb(current, total, filename):
                progress_bar.progress(current / total)
                status_text.text(f"Processing: {filename}")
            
            # Ingest
            result = await pipeline.ingest_directory(
                path,
                patterns=["*.md", "*.txt", "*.rst", "*.py", "*.yaml", "*.json"],
                progress_callback=progress_cb,
            )
            
            await pipeline.close()
            await provider.close()
            
            progress_bar.empty()
            status_text.empty()
            
            status.update(label="✅ Ingestion complete!", state="complete")
            
            st.success(
                f"Ingested **{result['successful']}** documents | "
                f"**{result['total_chunks']}** chunks | "
                f"**{result['total_questions']}** test questions"
            )
            
            # Refresh stats
            await refresh_knowledge_stats(db_url)
            
        except Exception as e:
            status.update(label="❌ Ingestion failed", state="error")
            st.error(f"Error: {e}")


async def evaluate_knowledge(db_url: str):
    """Run on-demand knowledge evaluation."""
    model_name = st.session_state.get('selected_model', 'qwen2.5-coder:1.5b')
    
    with st.status("📊 Evaluating knowledge...", expanded=True) as status:
        try:
            provider = OllamaProvider(model=model_name)
            await provider.initialize()
            
            store = PostgresVectorStore(DatabaseConfig(connection_string=db_url))
            await store.initialize()
            
            # Create a simple embed function
            async def embed_fn(text):
                return await provider.embed(text)
            
            from omniscient_rag.search import HybridSearcher
            searcher = HybridSearcher(store, embed_fn)
            
            async def llm_fn(prompt):
                from omniscient_llm.models import GenerationRequest
                response = await provider.generate(GenerationRequest(prompt=prompt))
                return response.content
            
            scorer = KnowledgeScorer(store, searcher, llm_fn)
            score = await scorer.evaluate()
            
            await store.close()
            await provider.close()
            
            st.session_state['knowledge_score'] = score
            
            status.update(label="✅ Evaluation complete!", state="complete")
            
            # Display score
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{score.overall_score:.1f}%")
            with col2:
                st.metric("Retrieval Precision", f"{score.retrieval_precision:.1%}")
            with col3:
                st.metric("Answer Accuracy", f"{score.answer_accuracy:.1f}")
            with col4:
                st.metric("Coverage", f"{score.coverage_ratio:.1%}")
            
        except Exception as e:
            status.update(label="❌ Evaluation failed", state="error")
            st.error(f"Error: {e}")


async def query_knowledge_base(query: str, db_url: str):
    """Query the knowledge base with learning enhancement."""
    model_name = st.session_state.get('selected_model', 'qwen2.5-coder:1.5b')
    alpha = st.session_state.get('hybrid_alpha', 0.5)
    
    with st.spinner("🔍 Searching and generating answer..."):
        try:
            async with ResourceManager(db_url, model_name) as rm:
                # Initialize learning components if available
                memory = None
                injector = None
                learned_context = None
                
                if HAS_LEARNING:
                    try:
                        memory = KnowledgeMemory(rm.store, rm.embed)
                        await memory.initialize()
                        
                        injector = ContextInjector(
                            memory,
                            InjectionConfig(
                                max_facts=5,
                                max_reasoning_chains=2,
                                include_few_shot_reasoning=True,
                            )
                        )
                        
                        # Get learned knowledge for this query
                        learned_context = await injector.build_context(query)
                        st.session_state['learned_knowledge'] = learned_context
                    except Exception as e:
                        st.warning(f"Learning features unavailable: {e}")
                
                from omniscient_rag.search import HybridSearcher, SearchConfig
                searcher = HybridSearcher(
                    rm.store,
                    rm.embed,
                    SearchConfig(top_k=5, alpha=alpha),
                )
                
                results = await searcher.search(query)
                
                if not results:
                    st.warning("No relevant information found in the knowledge base.")
                    return
                
                # Generate answer using retrieved context + learned knowledge
                context_text = "\n\n".join([f"[{r.source}]\n{r.chunk.content}" for r in results])
                
                prompt = f"""
{learned_context if learned_context else ''}

## Context from Knowledge Base:
{context_text}

## Question:
{query}

## Instructions:
Based on the context and any learned knowledge above, provide a clear, accurate answer.
If the context doesn't contain enough information, say so.

Answer:"""
                
                answer = await rm.generate(prompt)
                
                # Store query info for feedback
                st.session_state['current_query'] = query
                st.session_state['current_answer'] = answer
                st.session_state['retrieved_chunks'] = results
                
                # Store results
                st.session_state['query_results'] = [
                    {
                        'source': r.source,
                        'content': r.chunk.content,
                        'score': r.combined_score,
                        'vector_score': r.vector_score,
                        'bm25_score': r.bm25_score,
                        'chunk_id': str(r.chunk.id),
                    }
                    for r in results
                ]
            
        except Exception as e:
            st.error(f"Query failed: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")


async def submit_feedback(db_url: str, rating: float, feedback_type: str, correction: Optional[str] = None):
    """Submit user feedback to improve the AI's learning."""
    current_query = st.session_state.get('current_query')
    current_answer = st.session_state.get('current_answer')
    retrieved_chunks = st.session_state.get('retrieved_chunks', [])
    
    if not current_query or not current_answer:
        st.warning("No active query to provide feedback on")
        return
    
    if not HAS_LEARNING:
        st.warning("Learning module not available")
        return
    
    with st.spinner("💾 Saving feedback and learning..."):
        try:
            model_name = st.session_state.get('selected_model', 'qwen2.5-coder:1.5b')
            
            async with ResourceManager(db_url, model_name) as rm:
                # Initialize learning components
                memory = KnowledgeMemory(rm.store, rm.embed)
                await memory.initialize()
                
                learner = FeedbackLearner(memory, rm.generate)
                
                # Map feedback type string to enum
                feedback_type_map = {
                    "positive": FeedbackType.POSITIVE,
                    "negative": FeedbackType.NEGATIVE,
                    "neutral": FeedbackType.NEUTRAL,
                    "correction": FeedbackType.CORRECTION,
                }
                
                # Process feedback and learn
                learned = await learner.process_feedback(
                    question=current_query,
                    answer=current_answer,
                    rating=rating,
                    feedback_type=feedback_type_map.get(feedback_type, FeedbackType.NEUTRAL),
                    correction=correction,
                    retrieved_chunks=retrieved_chunks,
                )
                
                # Show what was learned
                st.success("✅ Feedback submitted successfully!")
                
                learned_summary = []
                if learned.get('facts_extracted', 0) > 0:
                    learned_summary.append(f"📝 Extracted {learned['facts_extracted']} new fact(s)")
                if learned.get('reasoning_stored'):
                    learned_summary.append("🧠 Stored reasoning pattern")
                if learned.get('refinement_learned'):
                    learned_summary.append("🔄 Learned query refinement")
                
                if learned_summary:
                    st.info("**What I learned:**\n" + "\n".join(f"- {item}" for item in learned_summary))
            
        except Exception as e:
            st.error(f"Failed to process feedback: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")


async def view_learned_facts(db_url: str):
    """View facts the AI has learned."""
    if not HAS_LEARNING:
        return
    
    with st.spinner("📚 Loading learned facts..."):
        try:
            store = PostgresVectorStore(DatabaseConfig(connection_string=db_url))
            await store.initialize()
            
            memory = KnowledgeMemory(store, None)
            await memory.initialize()
            
            # Get all facts ordered by confidence and usage
            async with store._pool.connection() as conn:
                cursor = await conn.execute(f"""
                    SELECT topic, fact, confidence, usage_count, success_rate, created_at
                    FROM {store.config.schema}.learned_facts
                    ORDER BY confidence DESC, usage_count DESC
                    LIMIT 50
                """)
                rows = await cursor.fetchall()
            
            await store.close()
            
            st.session_state['learned_facts'] = [
                {
                    'topic': row[0],
                    'fact': row[1],
                    'confidence': row[2],
                    'usage_count': row[3],
                    'success_rate': row[4],
                }
                for row in rows
            ]
            
        except Exception as e:
            st.error(f"Failed to load learned facts: {e}")


async def view_query_refinements(db_url: str):
    """View query refinement patterns the AI has learned."""
    if not HAS_LEARNING:
        return
    
    with st.spinner("🔄 Loading query patterns..."):
        try:
            store = PostgresVectorStore(DatabaseConfig(connection_string=db_url))
            await store.initialize()
            
            memory = KnowledgeMemory(store, None)
            await memory.initialize()
            
            # Get query refinements ordered by success rate
            async with store._pool.connection() as conn:
                cursor = await conn.execute(f"""
                    SELECT original_query, refined_query, times_applied, success_rate
                    FROM {store.config.schema}.query_refinements
                    ORDER BY success_rate DESC, times_applied DESC
                    LIMIT 20
                """)
                rows = await cursor.fetchall()
            
            await store.close()
            
            st.session_state['query_refinements'] = [
                {
                    'original_query': row[0],
                    'refined_query': row[1],
                    'times_applied': row[2],
                    'success_rate': row[3],
                }
                for row in rows
            ]
            
        except Exception as e:
            st.error(f"Failed to load query refinements: {e}")


if __name__ == "__main__":
    main()
