# omniscient-agents

AI analysis agents for the Omniscient Architect platform.

## Installation

```bash
pip install omniscient-agents
```

## Available Agents

- **ArchitectureAgent**: Analyzes code architecture, patterns, and design quality
- **EfficiencyAgent**: Identifies performance bottlenecks and optimization opportunities
- **ReliabilityAgent**: Checks error handling, testing coverage, and resilience
- **AlignmentAgent**: Verifies alignment between code, docs, and requirements

## Usage

```python
from omniscient_agents import ArchitectureAgent, ReliabilityAgent
from omniscient_core import AnalysisConfig, RepositoryInfo, FileAnalysis
from langchain_ollama import ChatOllama

# Initialize LLM
llm = ChatOllama(model="codellama:7b-instruct")

# Create agent
agent = ArchitectureAgent(
    llm=llm,
    name="ArchitectureAgent",
    description="Analyzes code architecture",
    analysis_focus="architecture"
)

# Run analysis
files = [FileAnalysis(path="main.py", size=1024, language="Python")]
repo = RepositoryInfo(path="/path/to/repo")
result = await agent.analyze(files, repo)

print(result.findings)
print(result.recommendations)
```

## Custom Agents

Create custom agents by extending `BaseAIAgent`:

```python
from omniscient_core import BaseAIAgent, AgentResponse

class SecurityAgent(BaseAIAgent):
    def get_prompt_template(self) -> str:
        return "Analyze for security vulnerabilities..."
    
    async def analyze(self, files, repo_info) -> AgentResponse:
        # Custom analysis logic
        pass
```
