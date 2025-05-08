# Benchmarking LLM Heuristics for Constraint Satisfaction Problems

This repository contains code for evaluating Large Language Models (LLMs) as heuristic providers for Constraint Satisfaction Problems (CSPs). It allows benchmarking LLM-based variable and value selection approaches against classical heuristics.

## Overview

The project implements an enhanced CSP solver that leverages LLMs to make intelligent decisions during the search process:

- **Variable Selection**: Uses LLMs to decide which variable to assign next
- **Value Ordering**: Uses LLMs to determine the order in which to try values for a selected variable
- **Infeasibility Detection**: Allows LLMs to detect when a problem is unsolvable based on current assignments

## Getting Started

### Prerequisites

- Python 3.8+
- `python-constraint` library
- A running LLM server supporting OpenAI-compatible API at `http://localhost:8000`

### Installation

```bash
git clone https://github.com/yourusername/benchmarking-llm-heuristics.git
cd benchmarking-llm-heuristics
pip install -r requirements.txt
```

## Running Tests

The project uses a mock LLM system for testing to avoid requiring an actual LLM service during tests:

```bash
# Run tests with coverage report
pytest tests/ --cov=.

# Run a specific test file
pytest tests/test_heuristic.py

# Run a specific test method
pytest tests/test_heuristic.py::TestAdvancedHeuristicSolver::test_variable_selection
```

## GitHub Actions

This repository is configured with GitHub Actions to automatically run tests on push to main branch or when creating a pull request. The workflow:

1. Sets up Python
2. Installs dependencies
3. Runs tests with coverage reporting
4. Uploads coverage to Codecov (if configured)

## Usage

Run experiments on various graph problems:

```bash
python run_experiments.py --problem maxcut --nodes 10 --density 0.3 --instances 5 --output results
```

Available problem types:
- `maxcut`: Maximum Cut Problem
- `mis`: Maximum Independent Set
- `mvc`: Minimum Vertex Cover
- `graph_coloring`: Graph Coloring Problem

## License

This project is licensed under the MIT License - see the LICENSE file for details.

