# 🧠 Omniscient Architect - Elite-Level Code Review System

## Overview

The **Omniscient Architect** is an advanced, AI-powered code review system that combines the capabilities of a Senior Staff Engineer, Product Manager, and Strategic CTO. It performs forensic, multi-perspective analysis of codebases to identify strengths, weaknesses, and strategic opportunities.

## Features

### Three-Phase Analysis Protocol

#### Phase 1: Ingestion & Deconstruction
- **Code Scanning**: Analyzes file structure, languages, and complexity
- **Objective Analysis**: Deconstructs project objectives into technical components
- **Statistics Generation**: Provides comprehensive codebase metrics

#### Phase 2: Multi-Agent Simulation
The system simulates four specialist sub-agents:

- **Agent Alpha (Architecture)**: Reviews file structure, design patterns, and scalability
- **Agent Beta (Efficiency & Logic)**: Identifies complexity issues, redundant code, and performance bottlenecks
- **Agent Gamma (Reliability & Security)**: Examines error handling, edge cases, and security vulnerabilities
- **Agent Delta (Alignment)**: Validates that code achieves the stated business objectives

#### Phase 3: Strategic Gap Analysis
- Compares current state vs. ideal state
- Identifies critical gaps in architecture, testing, documentation
- Provides actionable recommendations

## Installation

### Requirements
- Python 3.7+
- GitHub token (optional, for higher rate limits)
- Ollama (for AI analysis)

### Install Dependencies
```bash
# Install required packages
pip install langchain ollama httpx structlog rich streamlit PyGitHub
```

### Setup Ollama (for AI Analysis)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a code analysis model
ollama pull codellama:7b-instruct
```

## Docker Deployment

The Omniscient Architect can be easily deployed using Docker for consistent environments and simplified setup.

### Containerized workflow (dev vs prod)

This repo ships with two Compose files:

- `docker-compose.yml` (production-friendly): builds the app image and runs without mounting source code.
- `docker-compose.dev.yml` (developer experience): mounts the project folder into the container for instant reloads.

Common commands (using the modern `docker compose` syntax):

```powershell
# Development: runs app + Ollama, mounts source into /app
docker compose -f docker-compose.dev.yml up --build -d

# Production-like: runs app + Ollama without host mounts
docker compose up --build -d

# Check health (Streamlit)
Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health | Select-Object StatusCode

# Tail logs
docker compose logs -f app
```

Notes:
- Streamlit is bound to 0.0.0.0:8501 in the container; the app is exposed on host port 8501.
- Healthchecks target `/_stcore/health` for reliable 200 responses.
- Ollama runs at `http://ollama:11434` inside the app container (mapped to `localhost:11434` on the host).

### Prerequisites
- Docker and Docker Compose installed on your system
- At least 8GB RAM recommended for AI model inference

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone https://github.com/moshesham/AI-Omniscient-Architect.git
cd AI-Omniscient-Architect

# Start the application with Ollama
docker-compose up -d

# View logs
docker-compose logs -f app
```

The application will be available at `http://localhost:8501`

### Manual Docker Setup

If you prefer to run components separately:

```bash
# 1. Start Ollama service
docker run -d --name ollama -p 11434:11434 ollama/ollama

# 2. Pull the AI model
docker exec ollama ollama pull codellama:7b-instruct

# 3. Build and run the application
docker build -t omniscient-architect .
docker run -d --name omniscient-architect-app \
	-p 8501:8501 \
	--link ollama \
	-e OLLAMA_HOST=http://ollama:11434 \
	omniscient-architect
```

### Docker Commands

```bash
# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose up --build

# View running containers
docker-compose ps

# Access container logs
docker-compose logs app
docker-compose logs ollama

# Clean up (removes volumes too)
docker-compose down -v
```

### Configuration

The Docker setup includes:
- **Ollama Container**: Runs the AI inference service on port 11434
- **App Container**: Runs the Streamlit web interface on port 8501
- **Volume Persistence**: AI models are cached in a Docker volume
- **Health Checks**: Automatic container health monitoring

### App configuration (config.yaml)

Use `config.yaml` at the repo root to tune analysis behavior. Environment variables override file values.

Supported keys:
- `max_file_size_mb` (int): Maximum file size to ingest (default 10).
- `max_files` (int): Cap on number of files scanned (default 1000).
- `include_patterns` (list): Glob patterns to include (e.g., `['*.py','*.ts']`).
- `exclude_patterns` (list): Folders/patterns to skip (e.g., `['.git','node_modules']`).
- `ollama_model` (str): Model name, e.g., `codellama:7b-instruct`.
- `analysis_depth` (str): One of `quick|standard|deep`.

Environment overrides:
- `OLLAMA_MODEL`, `MAX_FILE_SIZE_MB`, `MAX_FILES`, `ANALYSIS_DEPTH`.

### Agent selection (UI and CLI)

Agents are registered via a simple registry and can be enabled/disabled at runtime.

- In the Streamlit UI (sidebar), pick agents under “Agents”. Your selection updates the active run.
- From CLI, use `--agents` with a comma-separated list of keys:

```powershell
python omniscient_architect_ai.py . --agents architecture,efficiency,reliability,alignment
```

Available agent keys include: `architecture`, `efficiency`, `reliability`, `alignment`, `github_repository`.

## Usage

### Command Line Interface

```bash
# Analyze the current directory
python omniscient_architect.py .

# Analyze a specific repository
python omniscient_architect.py /path/to/repository
```

### Web Interface (Recommended)

The web interface provides an intuitive way to analyze GitHub repositories with a modern UI.

#### Quick Start
```bash
# Run the web application
python run_web_app.py

# Or directly with streamlit
streamlit run web_app.py
```

#### Features
- **Repository Input**: Enter GitHub repository URLs for analysis
- **Project Objectives**: Provide context for more relevant analysis
- **Real-time Analysis**: Watch AI agents analyze your code in real-time
- **Comprehensive Reports**: View detailed findings with confidence scores
- **Configuration Options**: Customize analysis depth and AI models

#### Web Interface Usage
1. **Start the App**: Run `python run_web_app.py`
2. **Configure Settings**: Set your GitHub token and analysis preferences in the sidebar
3. **Enter Repository**: Paste a GitHub repository URL
4. **Add Context**: Optionally describe the project's objectives
5. **Analyze**: Click "Analyze Repository" to start the AI analysis
6. **Review Results**: Explore the comprehensive analysis report

### With Project Objective

```bash
# Analyze against a specific objective
python omniscient_architect.py . --objective "Build a data analytics dashboard"

# More complex objective
python omniscient_architect.py /path/to/repo --objective "Create a scalable user authentication system with OAuth2 support and role-based access control"
```

### Save Report to File

```bash
# Save analysis to markdown file
python omniscient_architect.py . --output analysis_report.md

# Save with objective
python omniscient_architect.py . --objective "Your objective here" --output report.md
```

## Output Format

The tool generates a comprehensive report with the following sections:

### 1. 🎯 Executive Summary & Alignment Check
- **Project Understanding**: 2-sentence summary of the codebase
- **Goal Alignment Score (0-100%)**: Quantitative measure of objective alignment
- **Component Breakdown**: Status of each identified component

### 2. 💪 Strengths (With Evidence)
- Lists well-implemented features and patterns
- Provides file-level evidence for each strength
- Explains why each strength matters

### 3. ⚠️ Critical Review: Weaknesses & Adjustments
Grouped by category:
- **Efficiency**: Performance and complexity issues
- **Accuracy**: Logic and correctness problems
- **Reliability**: Error handling and security concerns

Each weakness includes:
- Description of the issue
- Specific location (file/function)
- Concrete fix recommendations

### 4. 🧠 The Strategist's Advisor
- **Scalability**: How to handle 100x growth
- **Future-Proofing**: Recommended next features
- **Broader Application**: Potential use cases and adaptations

## Example Report

```
================================================================================
🧠 OMNISCIENT ARCHITECT - CODE REVIEW REPORT
================================================================================

## 1. 🎯 Executive Summary & Alignment Check

### Project Understanding:
This repository contains 45 files across 3 languages (Python, Markdown, JSON). 
The codebase appears to be a Streamlit web application with 5,234 lines of code.

### Goal Alignment Score: 75%

### Component Breakdown:
	• Core Logic: Present (12 files)
	• Documentation: Present (8 files)
	• Testing: Missing
	• Configuration: Present (3 files)

## 2. 💪 Strengths (With Evidence)

**Strength:** Utility/helper modules present
**Evidence:** Identified by Agent Alpha (Architecture)
**Why it matters:** This demonstrates adherence to best practices

...
```

## Use Cases

### For Development Teams
- **Pre-Release Review**: Comprehensive analysis before major releases
- **Onboarding**: Help new team members understand codebase structure
- **Technical Debt**: Identify and prioritize refactoring opportunities

### For Solo Developers
- **Self-Review**: Get objective feedback on your code
- **Learning**: Understand best practices through automated analysis
- **Portfolio Projects**: Ensure code quality before sharing

### For Project Managers
- **Health Checks**: Regular codebase health assessments
- **Resource Planning**: Identify areas needing attention
- **Risk Assessment**: Spot potential issues early

### For Educators
- **Student Projects**: Provide consistent, detailed feedback
- **Code Reviews**: Teach code review best practices
- **Assignment Grading**: Automated initial assessment

## Advanced Features

### Language Support

Currently supports analysis for:
- Python, JavaScript, TypeScript
- Java, C++, C, Go, Rust
- Ruby, PHP
- HTML, CSS, SQL
- Markdown, JSON, YAML

### Complexity Analysis

For Python files, the tool calculates complexity based on:
- Control structures (if, for, while)
- Exception handling (try/except)
- Function and class definitions

### Intelligent Filtering

Automatically ignores:
- Version control directories (.git)
- Build artifacts (dist, build)
- Dependencies (node_modules, venv)
- IDE files (.idea, .vscode)
- Cache directories (__pycache__, .pytest_cache)

## Customization

### Extending Agent Analysis

The tool is designed to be easily extensible. Each agent method can be enhanced:

```python
def _agent_alpha_architecture(self, structure: Dict[str, Any]) -> AgentFindings:
		"""Customize architecture analysis here"""
		# Add your custom checks
		pass
```

### Adding New Languages

Update the `language_patterns` dictionary:

```python
self.language_patterns = {
		'.py': 'Python',
		'.rs': 'Rust',
		'.your_ext': 'YourLanguage',  # Add here
}
```

## Best Practices

### When to Run

- ✅ Before code reviews
- ✅ After major feature additions
- ✅ During project planning phases
- ✅ When evaluating third-party code
- ❌ Not for real-time development (use linters instead)

### Interpreting Results

- **Alignment Score 80-100%**: Excellent alignment with objectives
- **Alignment Score 60-79%**: Good but needs improvement
- **Alignment Score 40-59%**: Significant gaps exist
- **Alignment Score <40%**: Major rework needed

### Combining with Other Tools

The Omniscient Architect complements but doesn't replace:
- **Linters** (pylint, eslint): For style and syntax
- **Testing Frameworks** (pytest, jest): For functional correctness
- **Security Scanners** (bandit, snyk): For vulnerability detection
- **Code Coverage** (coverage.py): For test completeness

## Limitations

### Current Limitations

- **Static Analysis Only**: Doesn't execute code
- **Pattern-Based**: May miss context-specific issues
- **No NLP Integration**: Objective matching uses simple pattern matching
- **Sampling**: Complex analysis limited to sample of files for performance

### Not a Replacement For

- Human code review
- Comprehensive security audits
- Performance profiling
- Integration testing

## FAQ

**Q: Does it modify my code?**  
A: No, it's read-only. It only analyzes and reports.

**Q: How long does analysis take?**  
A: Typically 10-30 seconds for repositories with <100 files, longer for larger codebases.

**Q: Can I use it in CI/CD?**  
A: Yes! Save output to a file and parse results in your pipeline.

**Q: What Python version is required?**  
A: Python 3.7+ (uses dataclasses and type hints)

**Q: Does it send data anywhere?**  
A: No, all analysis is local. No network calls.

## Contributing

Contributions are welcome! Areas for enhancement:

1. **NLP Integration**: Better objective understanding
2. **More Languages**: Additional language support
3. **Visual Reports**: HTML/PDF output formats
4. **CI/CD Integration**: GitHub Actions, GitLab CI templates
5. **Diff Analysis**: Analyze only changed files

## License

MIT License - See repository LICENSE file

## Authors

Data Science Analytical Handbook Team

## Acknowledgments

Inspired by the need for comprehensive, automated code review in educational and professional settings.

---

**Ready to analyze your code?**


# Data Science Analytical Interview Preparation Handbook

This repository provides a comprehensive handbook designed to assist in preparing for data science analytical interviews, with a specific focus on Meta.

## Repository Content and Structure

This repository is organized into the following sections and files:

### 1. Handbook (Data-Science-Analytical-Interview-Preparation-Handbook.MD)

This is the core document providing a comprehensive guide to Meta's data science interview process. It covers:

* **Introduction:** Overview of Meta's interview process and values.
* **Foundational Knowledge & Skills:** In-depth review of essential topics including statistics, probability, SQL, and data analysis with Python/R.
* **Interview-Specific Preparation:** Guidance on tackling each stage of the interview process, from technical screens to behavioral questions.
* **Analytical Execution/Case Study Interview:** Techniques for data analysis, hypothesis generation, and communication.
* **Analytical Reasoning/Product Sense Interview:** Strategies for clarifying problems, developing product sense, defining metrics, and designing experiments.
* **Resources & Communities:** Curated list of learning materials, online communities, and helpful resources.

### 2. Statistics and Probability Examples (Statistics-Probability-Example-Questions.MD)

This file provides detailed examples and solutions to common statistics and probability questions encountered in data science interviews. It covers:

* **Descriptive Statistics:** Questions on measures of central tendency, variability, and data visualization.
* **Probability:** Questions on Bayes' theorem, probability distributions, and sampling techniques.
* **Inferential Statistics:** Questions on hypothesis testing, p-values, confidence intervals, and significance testing.
* **Statistical Modeling:** Questions on regression analysis, A/B testing, and experimentation.

### 3. SQL Example Problems (sql-example-problems.md, sql-example-problems.pdf)

This file presents a collection of complex SQL problems designed to simulate real-world scenarios at Meta. The problems cover a wide range of SQL concepts and techniques, including joins, aggregations, subqueries, window functions, and more.

### 4. Analytical Hands-On Projects (Analytical-HandsOn-Projects/overview.md)

This section provides practical, hands-on experience in data analysis using open-source tools and data. The projects are structured in Jupyter Notebooks and guide you through each step of the analysis process.

### 5. Key Insights and Tips (Key-Insights-Tips-Meta.md)

This document provides a concise summary of key insights and tips for navigating the Meta data science interview process.

### 6. Extra Review Problems (Extra-Review-Problems/*.md)

This section contains additional SQL and analytical problems, including advanced SQL patterns and techniques.

### 7. Future Features (Future_fetures.md)

This file outlines potential additions and improvements to the handbook in the future.

### 8. GitHub Pages

The GitHub Pages for this repository is available at: [GitHub Pages URL](https://moshesham.github.io/Data-Science-Analytical-Handbook/)

### 9. 🧠 Omniscient Architect - Code Review Tool (NEW!)

An elite-level AI code review system that performs forensic, multi-perspective analysis of codebases. This tool simulates four specialist sub-agents (Architecture, Efficiency, Reliability, and Alignment) to provide comprehensive code reviews.

**Quick Start:**
```bash
# Analyze this repository
python omniscient_architect.py . --objective "Your project goal"

# Save detailed report
python omniscient_architect.py . --output review.md
```

**Documentation:**
* **Quick Start Guide:** `QUICK_START.md` - Get started in 5 minutes
* **Full Documentation:** `OMNISCIENT_ARCHITECT_README.md` - Complete feature reference
* **Sample Report:** `SAMPLE_ANALYSIS_REPORT.md` - See example output

**Use Cases:**
- Portfolio project reviews before sharing with recruiters
- Self-assessment for interview preparation projects
- Learning best practices from code analysis
- Understanding codebase structure and quality

## How to Use This Material

1. **Start with the Handbook:** Begin by reading the `Data-Science-Analytical-Interview-Preparation-Handbook.MD` file to gain a comprehensive understanding of Meta's interview process and the key skills assessed.
2. **Review Foundational Knowledge:** Use the handbook and the `Statistics-Probability-Example-Questions.MD` file to review and strengthen your understanding of core statistical concepts.
3. **Practice SQL:** Work through the problems in the `sql-example-problems.md` or `sql-example-problems.pdf` file to hone your SQL skills.
4. **Work on Hands-On Projects:** Engage with the projects in the `Analytical-HandsOn-Projects` directory to apply your knowledge in practical scenarios.
5. **Prepare for Behavioral Questions:** Review the `Behavioral-Mock-Interview.MD` file and practice answering behavioral questions.
6. **Utilize the Resources:** Explore the curated list of resources in the handbook to further enhance your preparation.

By following this approach, you can effectively utilize the materials in this repository to prepare for your Meta data science analytical interview. Good luck!

## Contributing and Working Together

This handbook is a collaborative effort, and contributions are welcome! If you have suggestions, find errors, or want to add more content, please feel free to open an issue or submit a pull request.
