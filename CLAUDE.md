# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This repository (`latent-diffusion-`) is in early setup — currently it contains only a README. No source code, dependencies, tests, or build configuration have been added yet.

## Project Intent

Based on the repository name, this is intended to be a latent diffusion project. Latent diffusion models (LDMs) perform diffusion in a compressed latent space rather than pixel space, making them more computationally efficient. The canonical reference implementation is [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion).

## Getting Started (once code is added)

When source code is added to this repository, update this file with:

- **Build/install commands** (e.g., `pip install -e .` or `conda env create -f environment.yaml`)
- **How to run training** (e.g., `python main.py --base configs/... -t`)
- **How to run inference/sampling**
- **How to run tests** (e.g., `pytest tests/` or a single test with `pytest tests/test_foo.py::test_bar`)
- **Lint/format commands** (e.g., `flake8`, `black`, `ruff`)
- **Architecture overview** covering model components, config system, and data pipeline
