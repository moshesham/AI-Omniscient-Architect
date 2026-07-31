# Release Guide

This repository ships in two distinct ways:

1. **Python packages** published to PyPI
2. **Application container images** published for Docker-based deployment

## Packaging strategy

- The repository root publishes `omniscient-architect` as a **meta-package**.
- Installable source packages live under `/packages/*`.
- Internal package dependencies use the same minor version line and are expected to be released together.

## Supported Python version

- Python 3.10+

## Release order

Publish dependent packages in this order when a coordinated release is required:

1. `omniscient-core`
2. `omniscient-llm`
3. `omniscient-tools`
4. `omniscient-github`
5. `omniscient-agents`
6. `omniscient-api`
7. `omniscient-rag`
8. `omniscient-architect`

## Local validation checklist

```bash
pip install -r requirements-dev.txt
pip install -e packages/core -e packages/llm -e packages/tools -e packages/github -e packages/agents -e packages/api -e packages/rag
pytest -q
python -m build
for pkg in core llm tools github agents api rag; do python -m build "packages/$pkg"; done
twine check dist/* packages/*/dist/*
```

## Canonical publishing workflow

- Workflow file: `.github/workflows/publish.yml`
- Publishing method: **PyPI trusted publishing with GitHub OIDC** (preferred) or `PYPI_API_TOKEN` GitHub secret fallback
- Manual dry runs are supported through `workflow_dispatch`

### Tag formats

- Meta-package release: `v0.2.0`
- Sub-package release: `omniscient-core-v0.2.0`

## PyPI trusted publisher setup

Configure PyPI to trust this GitHub repository/workflow instead of storing a long-lived API token:

1. Create each PyPI project (`omniscient-core`, `omniscient-llm`, `omniscient-agents`, `omniscient-tools`, `omniscient-github`, `omniscient-api`, `omniscient-rag`, `omniscient-architect`)
2. In PyPI, add a **trusted publisher**
3. Use:
   - **Owner**: `moshesham`
   - **Repository**: `AI-Omniscient-Architect`
   - **Workflow**: `publish.yml`
   - **Environment**: `pypi`
4. Protect the GitHub `pypi` environment as needed before allowing publish jobs

## PyPI API token fallback

If trusted publishing is not available yet, the workflow also supports a GitHub secret named `PYPI_API_TOKEN`.

1. Add `PYPI_API_TOKEN` as a repository secret or `pypi` environment secret in GitHub
2. Keep the workflow file unchanged; the publish job automatically prefers the secret when present
3. Never commit the token to the repository, workflow YAML, or documentation

## Container publishing workflow

- Workflow file: `.github/workflows/container-publish.yml`
- Registry target: `ghcr.io/<owner>/ai-omniscient-architect`
- Triggered from `main`, version tags, or manual runs

This workflow publishes the deployable application image. Environment-specific deployment should be handled by a downstream system after image publication.

## Rollback

- **PyPI package issue**: publish a corrected patch release; do not overwrite an existing release
- **Container issue**: redeploy the previous known-good image tag from GHCR
- **Documentation issue**: update `CHANGELOG.md` and release notes in a follow-up patch release
