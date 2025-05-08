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

Additional parameters:
- `--nodes`: Number of nodes in the graph
- `--density`: Edge density
- `--instances`: Number of instances to generate and solve
- `--node-limit`: Maximum number of nodes to explore (default: 100)
- `--time-limit`: Maximum solving time in seconds (default: 1800)

## Components

- `heuristic.py`: LLM-enhanced CSP solver implementation
- `run_experiments.py`: Experiment execution and evaluation
- `graph_instance_generator.py`: Problem instance generation

## Results

Experiment results are saved in the `results/` directory as JSON files containing:
- Problem details
- Solver performance metrics
- LLM vs. classical solver comparison
- Statistics on LLM guidance quality
- Infeasibility explanations (when detected)

## Citation

If you use this code in your research, please cite:

