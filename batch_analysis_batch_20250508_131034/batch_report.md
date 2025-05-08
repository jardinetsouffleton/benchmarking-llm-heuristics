# Batch Analysis Report: batch_20250508_131034

Generated on: 2025-05-08 20:16:12

## Overall Summary

- Total Instances: 5
- LLM Solver Success Rate: 40.00%
- Classical Solver Success Rate: 100.00%
- Average LLM Solve Time: 12670.62 seconds
- Average Classical Solve Time: 0.00 seconds

## Performance by Problem Type

| Problem Type | Count | LLM Success | Classical Success | LLM Avg Time (s) | Classical Avg Time (s) | LLM Avg Calls |
|-------------|-------|-------------|-------------------|------------------|------------------------|---------------|
| maxcut | 4 | 50.00% | 100.00% | 13905.47 | 0.00 | 78.75 |
| mis | 1 | 0.00% | 100.00% | 7731.22 | 0.00 | 41.00 |

## Problem Type: maxcut

### Success and Timing

- Instances: 4
- LLM Success Rate: 50.00%
- Classical Success Rate: 100.00%
- LLM Average Solve Time: 13905.47s (median: 14166.31s)
- Classical Average Solve Time: 0.00s (median: 0.00s)

### Solver Speed Comparison

- Average Time Difference (Classical - LLM): -10381.32s
- LLM Faster: 0 instances (0.00%)
- Classical Faster: 2 instances (50.00%)
- Similar Speed: 0 instances (0.00%)

### Solution Quality Comparison

- Comparable Solutions: 2 instances
- LLM Better Solutions: 2 (100.00%)
- Classical Better Solutions: 0 (0.00%)
- Equal Quality Solutions: 0 (0.00%)
- Average Score Difference: 6.00
- Max LLM Advantage: 9.00
- Max Classical Advantage: 0.00

### LLM Performance Metrics

- Average LLM Calls: 78.75
- Average Nodes Explored: 52.50
- Average Backtracks: 0.00
- Average Constraint Violations: 0.00
- Average Objective Improvements: 8.75

### Visualizations

![Success Rate](maxcut/success_rate.png)

![Solve Time Distribution](maxcut/solve_time_boxplot.png)

![Solution Quality](maxcut/solution_quality_pie.png)

![Score Comparison](maxcut/solution_score_comparison.png)

![LLM Calls Distribution](maxcut/llm_calls_hist.png)

![LLM Calls vs Time](maxcut/llm_calls_vs_time.png)


## Problem Type: mis

### Success and Timing

- Instances: 1
- LLM Success Rate: 0.00%
- Classical Success Rate: 100.00%
- LLM Average Solve Time: 7731.22s (median: 7731.22s)
- Classical Average Solve Time: 0.00s (median: 0.00s)

### Solver Speed Comparison

- Average Time Difference (Classical - LLM): 0.00s
- LLM Faster: 0 instances (0.00%)
- Classical Faster: 0 instances (0.00%)
- Similar Speed: 0 instances (0.00%)

### Solution Quality Comparison

- Comparable Solutions: 0 instances

### LLM Performance Metrics

- Average LLM Calls: 41.00
- Average Nodes Explored: 0.00
- Average Backtracks: 0.00
- Average Constraint Violations: 0.00
- Average Objective Improvements: 3.00

### Visualizations

![Success Rate](mis/success_rate.png)

![Solve Time Distribution](mis/solve_time_boxplot.png)

![LLM Calls Distribution](mis/llm_calls_hist.png)

![LLM Calls vs Time](mis/llm_calls_vs_time.png)


## Batch-wide Visualizations

![Success by Problem Type](success_by_problem_type.png)

![Solve Time by Problem Type](solve_time_by_problem_type.png)

![Quality Winner by Problem Type](quality_winner_by_problem_type.png)

![LLM Calls by Problem Type](llm_calls_by_problem_type.png)

