# Changelog

All notable changes to this repository are documented in this file.

## [0.2.0] - 2026-07-30

### Changed
- Standardized the repository around a root meta-package plus independently published sub-packages.
- Normalized package metadata, Python support, and internal dependency ranges across the monorepo.
- Split Streamlit app runtime requirements from development/build requirements.
- Repaired the PyPI publishing workflow and aligned it with trusted publishing via GitHub OIDC.
- Expanded CI to validate builds, package metadata, and installation from built artifacts.
- Added a dedicated container image publishing workflow for the Docker deployment track.

### Documentation
- Added a consolidated release guide covering packaging, tags, publishing, and rollback.
- Clarified that application deployment and Python package publication are separate release tracks.
