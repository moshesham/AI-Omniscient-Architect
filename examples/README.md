# Examples

This directory contains example scripts demonstrating how to use the Omniscient Architect tool.

## analyze_self.py

A Python script that analyzes the Omniscient Architect repository itself using the CLI interface.

### What it does

- Downloads and analyzes the codebase of this repository
- Uses AI agents to provide insights on architecture, efficiency, reliability, and goal alignment
- Saves the comprehensive report to `examples/analysis_report.md`

### Running the example

1. Ensure Ollama is running with the required model:
   ```bash
   ollama pull codellama:7b-instruct
   ```

2. Run the example:
   ```bash
   python examples/analyze_self.py
   ```

3. View the generated report:
   ```bash
   cat examples/analysis_report.md
   ```

### Customizing the example

You can modify the script to analyze different repositories by changing the GitHub URL and objective in the `sys.argv` list.

For example, to analyze a different repo:
```python
sys.argv = [
    "omniscient_architect_ai.py",
    "https://github.com/your-org/your-repo",
    "--objective",
    "Your custom objective here",
    "--output",
    "examples/custom_report.md"
]
```

### Web Interface Alternative

For a more interactive experience, use the Streamlit web interface:

```bash
# Start the web app
python -m streamlit run web_app.py

# Then visit http://localhost:8501 and enter the repo URL
```