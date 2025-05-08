#!/usr/bin/env python3
"""
Aggregate results from CSP experiment batches.

This script analyzes a single batch of experiments, providing detailed
results per problem type with statistics and visualizations.
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

def find_batch_directories(results_dir="results"):
    """Find all batch directories in the results directory."""
    batch_dirs = []
    for dir_path in glob.glob(os.path.join(results_dir, "batch_*")):
        if os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "batch_config.json")):
            batch_dirs.append(dir_path)
    return sorted(batch_dirs)

def load_batch_data(batch_dir):
    """Load all relevant data from a batch directory."""
    batch_data = {
        "dir": batch_dir,
        "batch_id": os.path.basename(batch_dir),
    }
    
    # Load batch configuration
    config_path = os.path.join(batch_dir, "batch_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            batch_data["config"] = json.load(f)
    
    # Load batch summary
    summary_path = os.path.join(batch_dir, "batch_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            batch_data["summary"] = json.load(f)
    
    # Load batch stats
    stats_path = os.path.join(batch_dir, "batch_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            batch_data["stats"] = json.load(f)
    
    # Find and load all result files for each problem type
    batch_data["results"] = []
    
    # Check if we have problem types defined in the config
    problem_types = batch_data.get("config", {}).get("problem_types", [])
    
    # Track completed runs for accurate success rate calculation
    completed_runs = 0
    successful_llm_runs = 0
    successful_classical_runs = 0
    
    # If no problem types in config, scan for problem-specific JSON files directly
    if not problem_types:
        for result_file in glob.glob(os.path.join(batch_dir, "*.json")):
            # Skip config, summary, stats files, and instance_*.json files
            filename = os.path.basename(result_file)
            if (any(x in filename for x in ["batch_config", "batch_summary", "batch_stats"]) or
                "instance_" in filename):
                continue
            
            # Only process results with format {problem name}_{date}_{time}.json
            if not any(pt in filename for pt in ["maxcut", "mis", "mvc", "graph_coloring"]):
                continue
                
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    # Add source file
                    result_data["source_file"] = result_file
                    batch_data["results"].append(result_data)
                    
                    # Count completed runs and successes
                    completed_runs += 1
                    if result_data.get("llm_solver", {}).get("success", False):
                        successful_llm_runs += 1
                    if result_data.get("classical_solver", {}).get("success", False):
                        successful_classical_runs += 1
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    else:
        # Look in problem-specific directories
        for problem_type in problem_types:
            problem_dir = os.path.join(batch_dir, problem_type)
            if os.path.isdir(problem_dir):
                # Look for result files with the pattern {problem type}_{date}_{time}.json
                result_pattern = os.path.join(problem_dir, f"{problem_type}_*.json")
                for result_file in glob.glob(result_pattern):
                    # Skip instance_*.json files
                    if "instance_" in os.path.basename(result_file):
                        continue
                        
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                            # Add source file
                            result_data["source_file"] = result_file
                            
                            # If problem_type isn't in the data, add it based on the directory
                            if "problem_type" not in result_data:
                                result_data["problem_type"] = problem_type
                                
                            batch_data["results"].append(result_data)
                            
                            # Count completed runs and successes
                            completed_runs += 1
                            if result_data.get("llm_solver", {}).get("success", False):
                                successful_llm_runs += 1
                            if result_data.get("classical_solver", {}).get("success", False):
                                successful_classical_runs += 1
                    except Exception as e:
                        print(f"Error loading {result_file}: {e}")
    
    # If no results were found through the directed search, do a recursive search
    if not batch_data["results"]:
        print(f"No results found in expected locations, performing deeper search in {batch_dir}")
        for root, _, files in os.walk(batch_dir):
            for file in files:
                # Only process result files with format {problem name}_{date}_{time}.json
                if (file.endswith('.json') and 
                    not "instance_" in file and
                    not any(x in file for x in ["batch_config", "batch_summary", "batch_stats"]) and
                    any(pt in file for pt in ["maxcut", "mis", "mvc", "graph_coloring"])):
                    
                    result_file = os.path.join(root, file)
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                            
                            # Check if this is a result file (has llm_solver or classical_solver fields)
                            if "llm_solver" in result_data or "classical_solver" in result_data:
                                # Add source file
                                result_data["source_file"] = result_file
                                
                                # If problem_type isn't in the data, try to determine from filename or parent dir
                                if "problem_type" not in result_data:
                                    parent_dir = os.path.basename(os.path.dirname(result_file))
                                    # Check if parent directory is a problem type
                                    if parent_dir in ["maxcut", "mis", "mvc", "graph_coloring"]:
                                        result_data["problem_type"] = parent_dir
                                    # Otherwise try to extract from filename (e.g., maxcut_20250508_131034.json)
                                    else:
                                        for pt in ["maxcut", "mis", "mvc", "graph_coloring"]:
                                            if pt in file:
                                                result_data["problem_type"] = pt
                                                break
                                            
                                batch_data["results"].append(result_data)
                                
                                # Count completed runs and successes
                                completed_runs += 1
                                if result_data.get("llm_solver", {}).get("success", False):
                                    successful_llm_runs += 1
                                if result_data.get("classical_solver", {}).get("success", False):
                                    successful_classical_runs += 1
                    except Exception as e:
                        print(f"Error loading {result_file}: {e}")
    
    # Store overall success rates based on completed runs
    batch_data["overall_stats"] = {
        "completed_runs": completed_runs,
        "successful_llm_runs": successful_llm_runs,
        "successful_classical_runs": successful_classical_runs,
        "llm_success_rate": (successful_llm_runs / completed_runs * 100) if completed_runs > 0 else 0,
        "classical_success_rate": (successful_classical_runs / completed_runs * 100) if completed_runs > 0 else 0
    }
    
    print(f"Loaded {len(batch_data['results'])} result files from {batch_dir}")
    print(f"Completed runs: {completed_runs}, LLM success rate: {batch_data['overall_stats']['llm_success_rate']:.2f}%, Classical success rate: {batch_data['overall_stats']['classical_success_rate']:.2f}%")
    return batch_data

def extract_metrics(batch_data):
    """Extract key metrics from batch data into a flat DataFrame."""
    rows = []
    
    # Extract configuration parameters that will be common to all rows
    config = batch_data.get("config", {})
    common = {
        "batch_id": batch_data["batch_id"],
        "timestamp": config.get("timestamp", ""),
        "nodes": config.get("nodes", 0),
        "density": config.get("density", 0),
        "node_limit": config.get("node_limit", 0),
        "time_limit": config.get("time_limit", 0),
    }
    
    # Process each result
    for result in batch_data.get("results", []):
        row = common.copy()
        
        # Extract basic problem information
        row.update({
            "problem_type": result.get("problem_type", "unknown"),
            "problem_id": result.get("problem_id", ""),
            "num_nodes": result.get("num_nodes", 0),
            "num_edges": result.get("num_edges", 0),
            "source_file": result.get("source_file", ""),
        })
        
        # Extract LLM solver metrics
        llm_solver = result.get("llm_solver", {})
        row.update({
            "llm_success": llm_solver.get("success", False),
            "llm_solve_time": llm_solver.get("solve_time", 0),
            "llm_solutions_found": llm_solver.get("solutions_found", 0),
            "llm_best_score": llm_solver.get("best_score", 0),
        })
        
        # Extract classical solver metrics
        classical_solver = result.get("classical_solver", {})
        row.update({
            "classical_success": classical_solver.get("success", False),
            "classical_solve_time": classical_solver.get("solve_time", 0),
            "classical_solutions_found": classical_solver.get("solutions_found", 0),
            "classical_best_score": classical_solver.get("best_score", 0),
        })
        
        # Extract comparative metrics
        comparative = result.get("comparative_analysis", {})
        row.update({
            "comparable": comparative.get("comparable", False),
            "score_difference": comparative.get("score_difference", 0),
            "score_ratio": comparative.get("score_ratio", 0),
            "time_difference": comparative.get("time_difference", 0),
            "better_quality": comparative.get("better_quality", ""),
            "faster_solver": comparative.get("faster_solver", ""),
        })
        
        # Extract LLM specific statistics if available
        solving_stats = llm_solver.get("solving_stats", {})
        row.update({
            "llm_nodes_explored": solving_stats.get("nodes_explored", 0),
            "llm_backtrack_count": solving_stats.get("backtrack_count", 0),
            "llm_calls": solving_stats.get("llm_calls", 0),
            "llm_constraint_violations": solving_stats.get("constraint_violations", 0),
            "llm_objective_improvements": solving_stats.get("objective_improvements", 0),
        })
        
        # Extract LLM call times if available
        llm_call_times = solving_stats.get("llm_call_times", [])
        if llm_call_times:
            row.update({
                "llm_avg_call_time": np.mean(llm_call_times),
                "llm_median_call_time": np.median(llm_call_times),
                "llm_min_call_time": min(llm_call_times),
                "llm_max_call_time": max(llm_call_times),
                "llm_total_call_time": sum(llm_call_times),
            })
        
        # Include detailed solution data for further analysis
        if "best_solution" in llm_solver:
            row["llm_best_solution"] = llm_solver["best_solution"]
        if "best_solution" in classical_solver:
            row["classical_best_solution"] = classical_solver["best_solution"]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ensure required columns exist
    required_cols = [
        "llm_success", "classical_success", 
        "llm_solve_time", "classical_solve_time",
        "llm_solutions_found", "classical_solutions_found",
        "llm_best_score", "classical_best_score",
        "comparable", "better_quality", "faster_solver"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    
    return df

def calculate_per_problem_metrics(df):
    """Calculate detailed metrics for each problem type."""
    problem_types = df["problem_type"].unique()
    
    metrics_by_problem = {}
    for problem_type in problem_types:
        problem_df = df[df["problem_type"] == problem_type]
        
        # Basic statistics
        stats = {
            "count": len(problem_df),
            "llm_success_rate": problem_df["llm_success"].mean() * 100,
            "classical_success_rate": problem_df["classical_success"].mean() * 100,
            "avg_llm_solve_time": problem_df["llm_solve_time"].mean(),
            "avg_classical_solve_time": problem_df["classical_solve_time"].mean(),
            "median_llm_solve_time": problem_df["llm_solve_time"].median(),
            "median_classical_solve_time": problem_df["classical_solve_time"].median(),
            "better_quality_counts": problem_df["better_quality"].value_counts().to_dict(),
            "faster_solver_counts": problem_df["faster_solver"].value_counts().to_dict(),
        }
        
        # LLM specific metrics
        stats.update({
            "avg_llm_calls": problem_df["llm_calls"].mean(),
            "avg_llm_nodes_explored": problem_df["llm_nodes_explored"].mean(),
            "avg_llm_backtrack_count": problem_df["llm_backtrack_count"].mean(),
            "avg_llm_objective_improvements": problem_df["llm_objective_improvements"].mean(),
            "avg_llm_constraint_violations": problem_df["llm_constraint_violations"].mean(),
        })
        
        # Solution scores statistics
        comparable_df = problem_df[problem_df["comparable"] == True]
        stats.update({
            "comparable_count": len(comparable_df),
            "llm_wins": len(comparable_df[comparable_df["better_quality"] == "LLM"]),
            "classical_wins": len(comparable_df[comparable_df["better_quality"] == "Classical"]),
            "ties": len(comparable_df[comparable_df["better_quality"] == "Tie"]),
        })
        
        if stats["comparable_count"] > 0:
            stats.update({
                "llm_win_rate": stats["llm_wins"] / stats["comparable_count"] * 100,
                "classical_win_rate": stats["classical_wins"] / stats["comparable_count"] * 100,
                "tie_rate": stats["ties"] / stats["comparable_count"] * 100,
                
                # Score differences when comparable
                "avg_score_difference": comparable_df["score_difference"].mean(),
                "median_score_difference": comparable_df["score_difference"].median(),
                "max_llm_advantage": comparable_df["score_difference"].max(),
                "max_classical_advantage": comparable_df["score_difference"].min() * -1 if comparable_df["score_difference"].min() < 0 else 0,
            })
        
        # Time performance
        stats.update({
            "avg_time_difference": problem_df["time_difference"].mean(),
            "llm_faster_count": len(problem_df[problem_df["faster_solver"] == "LLM"]),
            "classical_faster_count": len(problem_df[problem_df["faster_solver"] == "Classical"]),
            "same_speed_count": len(problem_df[problem_df["faster_solver"] == "Tie"]),
        })
        
        if stats["count"] > 0:
            stats.update({
                "llm_faster_rate": stats["llm_faster_count"] / stats["count"] * 100,
                "classical_faster_rate": stats["classical_faster_count"] / stats["count"] * 100,
                "same_speed_rate": stats["same_speed_count"] / stats["count"] * 100,
            })
        
        metrics_by_problem[problem_type] = stats
    
    # Overall metrics across all problem types
    overall = {
        "total_instances": len(df),
        "llm_success_rate": df["llm_success"].mean() * 100,
        "classical_success_rate": df["classical_success"].mean() * 100,
        "avg_llm_solve_time": df["llm_solve_time"].mean(),
        "avg_classical_solve_time": df["classical_solve_time"].mean(),
    }
    
    return {
        "by_problem": metrics_by_problem,
        "overall": overall,
    }

def generate_problem_visualizations(df, problem_type, output_dir):
    """Generate visualizations for a specific problem type."""
    problem_dir = os.path.join(output_dir, problem_type)
    os.makedirs(problem_dir, exist_ok=True)
    
    problem_df = df[df["problem_type"] == problem_type]
    
    # 1. Success rate comparison
    plt.figure(figsize=(10, 6))
    success_data = [
        problem_df["llm_success"].mean() * 100,
        problem_df["classical_success"].mean() * 100
    ]
    plt.bar(["LLM", "Classical"], success_data, color=['#ff9999', '#66b3ff'])
    plt.ylim(0, 100)
    plt.title(f'Success Rate - {problem_type}')
    plt.ylabel('Success Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(problem_dir, "success_rate.png"))
    plt.close()
    
    # 2. Solve time comparison as boxplot
    plt.figure(figsize=(10, 6))
    solve_time_df = problem_df.melt(
        id_vars=["problem_id"],
        value_vars=["llm_solve_time", "classical_solve_time"],
        var_name="solver",
        value_name="solve_time"
    )
    solve_time_df["solver"] = solve_time_df["solver"].map({
        "llm_solve_time": "LLM", 
        "classical_solve_time": "Classical"
    })
    
    # Filter extreme outliers
    upper_limit = solve_time_df["solve_time"].quantile(0.95)
    filtered_df = solve_time_df[solve_time_df["solve_time"] <= upper_limit]
    
    sns.boxplot(x="solver", y="solve_time", data=filtered_df)
    plt.title(f'Solve Time Distribution - {problem_type}')
    plt.ylabel('Solve Time (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(problem_dir, "solve_time_boxplot.png"))
    plt.close()
    
    # 3. Solution quality comparison (when comparable)
    comparable_df = problem_df[problem_df["comparable"] == True]
    if len(comparable_df) > 0:
        plt.figure(figsize=(8, 8))
        quality_data = [
            len(comparable_df[comparable_df["better_quality"] == "LLM"]),
            len(comparable_df[comparable_df["better_quality"] == "Classical"]),
            len(comparable_df[comparable_df["better_quality"] == "Tie"])
        ]
        labels = ['LLM Better', 'Classical Better', 'Equal Quality']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        plt.pie(quality_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'Solution Quality Comparison - {problem_type}')
        plt.tight_layout()
        plt.savefig(os.path.join(problem_dir, "solution_quality_pie.png"))
        plt.close()
    
    # 4. LLM calls distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(problem_df["llm_calls"], kde=True, bins=15)
    plt.title(f'Distribution of LLM API Calls - {problem_type}')
    plt.xlabel('Number of LLM Calls')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(problem_dir, "llm_calls_hist.png"))
    plt.close()
    
    # 5. Correlation between LLM calls and solve time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="llm_calls", y="llm_solve_time", data=problem_df)
    plt.title(f'LLM Calls vs. Solve Time - {problem_type}')
    plt.xlabel('Number of LLM Calls')
    plt.ylabel('Solve Time (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(problem_dir, "llm_calls_vs_time.png"))
    plt.close()
    
    # 6. If there's solution score data, plot a score comparison
    if "llm_best_score" in problem_df.columns and "classical_best_score" in problem_df.columns:
        comparable_df = problem_df[problem_df["comparable"] == True]
        if len(comparable_df) > 0:
            plt.figure(figsize=(10, 6))
            
            # For maximization problems (higher is better)
            if problem_type in ["maxcut", "mis"]:
                sns.scatterplot(x="classical_best_score", y="llm_best_score", data=comparable_df)
                max_score = max(
                    comparable_df["llm_best_score"].max(),
                    comparable_df["classical_best_score"].max()
                )
                plt.plot([0, max_score], [0, max_score], 'k--', alpha=0.5)  # identity line
                plt.title(f'Solution Quality Comparison - {problem_type} (higher is better)')
            
            # For minimization problems (lower is better)
            elif problem_type in ["mvc", "graph_coloring"]:
                sns.scatterplot(x="classical_best_score", y="llm_best_score", data=comparable_df)
                max_score = max(
                    comparable_df["llm_best_score"].max(),
                    comparable_df["classical_best_score"].max()
                )
                plt.plot([0, max_score], [0, max_score], 'k--', alpha=0.5)  # identity line
                plt.title(f'Solution Quality Comparison - {problem_type} (lower is better)')
            
            plt.xlabel('Classical Solution Score')
            plt.ylabel('LLM Solution Score')
            plt.tight_layout()
            plt.savefig(os.path.join(problem_dir, "solution_score_comparison.png"))
            plt.close()
    
    return problem_dir

def generate_batch_visualizations(df, metrics, output_dir):
    """Generate overview visualizations for the entire batch."""
    # 1. Success rates by problem type
    plt.figure(figsize=(12, 6))
    problem_types = df["problem_type"].unique()
    x = np.arange(len(problem_types))
    width = 0.35
    
    llm_success = [df[df["problem_type"] == pt]["llm_success"].mean() * 100 for pt in problem_types]
    classical_success = [df[df["problem_type"] == pt]["classical_success"].mean() * 100 for pt in problem_types]
    
    plt.bar(x - width/2, llm_success, width, label='LLM')
    plt.bar(x + width/2, classical_success, width, label='Classical')
    
    plt.xlabel('Problem Type')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by Problem Type')
    plt.xticks(x, problem_types)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_by_problem_type.png"))
    plt.close()
    
    # 2. Average solve time by problem type
    plt.figure(figsize=(12, 6))
    
    llm_times = [df[df["problem_type"] == pt]["llm_solve_time"].mean() for pt in problem_types]
    classical_times = [df[df["problem_type"] == pt]["classical_solve_time"].mean() for pt in problem_types]
    
    plt.bar(x - width/2, llm_times, width, label='LLM')
    plt.bar(x + width/2, classical_times, width, label='Classical')
    
    plt.xlabel('Problem Type')
    plt.ylabel('Average Solve Time (s)')
    plt.title('Average Solve Time by Problem Type')
    plt.xticks(x, problem_types)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "solve_time_by_problem_type.png"))
    plt.close()
    
    # 3. Solution quality winner distribution across problem types
    comparable_df = df[df["comparable"] == True]
    if len(comparable_df) > 0:
        quality_data = []
        for pt in problem_types:
            pt_df = comparable_df[comparable_df["problem_type"] == pt]
            if len(pt_df) > 0:
                quality_data.append({
                    'problem_type': pt,
                    'LLM Better': len(pt_df[pt_df["better_quality"] == "LLM"]) / len(pt_df) * 100,
                    'Classical Better': len(pt_df[pt_df["better_quality"] == "Classical"]) / len(pt_df) * 100,
                    'Equal Quality': len(pt_df[pt_df["better_quality"] == "Tie"]) / len(pt_df) * 100
                })
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            quality_df = quality_df.melt(
                id_vars=['problem_type'],
                value_vars=['LLM Better', 'Classical Better', 'Equal Quality'],
                var_name='Winner',
                value_name='Percentage'
            )
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='problem_type', y='Percentage', hue='Winner', data=quality_df)
            plt.xlabel('Problem Type')
            plt.ylabel('Percentage (%)')
            plt.title('Solution Quality Winner Distribution by Problem Type')
            plt.legend(title='Winner')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "quality_winner_by_problem_type.png"))
            plt.close()
    
    # 4. LLM calls by problem type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="problem_type", y="llm_calls", data=df)
    plt.xlabel('Problem Type')
    plt.ylabel('Number of LLM Calls')
    plt.title('LLM Calls by Problem Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "llm_calls_by_problem_type.png"))
    plt.close()
    
    return True

def generate_detailed_report(metrics, problem_dirs, output_dir, batch_id):
    """Generate a detailed markdown report with results per problem type."""
    report_path = os.path.join(output_dir, "batch_report.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Batch Analysis Report: {batch_id}\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        f.write(f"- Total Instances: {metrics['overall']['total_instances']}\n")
        f.write(f"- LLM Solver Success Rate: {metrics['overall']['llm_success_rate']:.2f}%\n")
        f.write(f"- Classical Solver Success Rate: {metrics['overall']['classical_success_rate']:.2f}%\n")
        f.write(f"- Average LLM Solve Time: {metrics['overall']['avg_llm_solve_time']:.2f} seconds\n")
        f.write(f"- Average Classical Solve Time: {metrics['overall']['avg_classical_solve_time']:.2f} seconds\n\n")
        
        # Performance summary table by problem type
        f.write("## Performance by Problem Type\n\n")
        f.write("| Problem Type | Count | LLM Success | Classical Success | LLM Avg Time (s) | Classical Avg Time (s) | LLM Avg Calls |\n")
        f.write("|-------------|-------|-------------|-------------------|------------------|------------------------|---------------|\n")
        
        for problem_type, stats in metrics["by_problem"].items():
            f.write(f"| {problem_type} | {stats['count']} | {stats['llm_success_rate']:.2f}% | {stats['classical_success_rate']:.2f}% | {stats['avg_llm_solve_time']:.2f} | {stats['avg_classical_solve_time']:.2f} | {stats['avg_llm_calls']:.2f} |\n")
        
        # Detailed sections for each problem type
        for problem_type, stats in metrics["by_problem"].items():
            f.write(f"\n## Problem Type: {problem_type}\n\n")
            
            # Success and timing
            f.write("### Success and Timing\n\n")
            f.write(f"- Instances: {stats['count']}\n")
            f.write(f"- LLM Success Rate: {stats['llm_success_rate']:.2f}%\n")
            f.write(f"- Classical Success Rate: {stats['classical_success_rate']:.2f}%\n")
            f.write(f"- LLM Average Solve Time: {stats['avg_llm_solve_time']:.2f}s (median: {stats['median_llm_solve_time']:.2f}s)\n")
            f.write(f"- Classical Average Solve Time: {stats['avg_classical_solve_time']:.2f}s (median: {stats['median_classical_solve_time']:.2f}s)\n")
            
            # Solver speed comparison
            f.write("\n### Solver Speed Comparison\n\n")
            f.write(f"- Average Time Difference (Classical - LLM): {stats['avg_time_difference']:.2f}s\n")
            f.write(f"- LLM Faster: {stats.get('llm_faster_count', 0)} instances ({stats.get('llm_faster_rate', 0):.2f}%)\n")
            f.write(f"- Classical Faster: {stats.get('classical_faster_count', 0)} instances ({stats.get('classical_faster_rate', 0):.2f}%)\n")
            f.write(f"- Similar Speed: {stats.get('same_speed_count', 0)} instances ({stats.get('same_speed_rate', 0):.2f}%)\n")
            
            # Solution quality comparison
            f.write("\n### Solution Quality Comparison\n\n")
            f.write(f"- Comparable Solutions: {stats['comparable_count']} instances\n")
            
            if stats['comparable_count'] > 0:
                f.write(f"- LLM Better Solutions: {stats['llm_wins']} ({stats['llm_win_rate']:.2f}%)\n")
                f.write(f"- Classical Better Solutions: {stats['classical_wins']} ({stats['classical_win_rate']:.2f}%)\n")
                f.write(f"- Equal Quality Solutions: {stats['ties']} ({stats['tie_rate']:.2f}%)\n")
                f.write(f"- Average Score Difference: {stats['avg_score_difference']:.2f}\n")
                f.write(f"- Max LLM Advantage: {stats['max_llm_advantage']:.2f}\n")
                f.write(f"- Max Classical Advantage: {stats['max_classical_advantage']:.2f}\n")
            
            # LLM behavior
            f.write("\n### LLM Performance Metrics\n\n")
            f.write(f"- Average LLM Calls: {stats['avg_llm_calls']:.2f}\n")
            f.write(f"- Average Nodes Explored: {stats['avg_llm_nodes_explored']:.2f}\n")
            f.write(f"- Average Backtracks: {stats['avg_llm_backtrack_count']:.2f}\n")
            f.write(f"- Average Constraint Violations: {stats['avg_llm_constraint_violations']:.2f}\n")
            f.write(f"- Average Objective Improvements: {stats['avg_llm_objective_improvements']:.2f}\n")
            
            # Visualizations
            if problem_type in problem_dirs:
                f.write("\n### Visualizations\n\n")
                
                # Make paths relative to the report
                problem_rel_dir = os.path.relpath(problem_dirs[problem_type], output_dir)
                
                f.write(f"![Success Rate]({os.path.join(problem_rel_dir, 'success_rate.png')})\n\n")
                f.write(f"![Solve Time Distribution]({os.path.join(problem_rel_dir, 'solve_time_boxplot.png')})\n\n")
                
                # Only include pie chart if we have comparable solutions
                if stats['comparable_count'] > 0:
                    f.write(f"![Solution Quality]({os.path.join(problem_rel_dir, 'solution_quality_pie.png')})\n\n")
                    
                    # Include score comparison if it exists
                    if os.path.exists(os.path.join(problem_dirs[problem_type], "solution_score_comparison.png")):
                        f.write(f"![Score Comparison]({os.path.join(problem_rel_dir, 'solution_score_comparison.png')})\n\n")
                
                f.write(f"![LLM Calls Distribution]({os.path.join(problem_rel_dir, 'llm_calls_hist.png')})\n\n")
                f.write(f"![LLM Calls vs Time]({os.path.join(problem_rel_dir, 'llm_calls_vs_time.png')})\n\n")
        
        # Overall visualizations
        f.write("\n## Batch-wide Visualizations\n\n")
        f.write("![Success by Problem Type](success_by_problem_type.png)\n\n")
        f.write("![Solve Time by Problem Type](solve_time_by_problem_type.png)\n\n")
        f.write("![Quality Winner by Problem Type](quality_winner_by_problem_type.png)\n\n")
        f.write("![LLM Calls by Problem Type](llm_calls_by_problem_type.png)\n\n")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Analyze a single batch of CSP experiments")
    parser.add_argument("--batch-dir", required=True, help="Path to the batch directory to analyze")
    parser.add_argument("--output-dir", default=None, help="Directory to save analysis results (defaults to batch_analysis_<batchid>)")
    args = parser.parse_args()
    
    # Validate the batch directory
    if not os.path.isdir(args.batch_dir):
        print(f"Error: Batch directory {args.batch_dir} does not exist")
        return
    
    batch_id = os.path.basename(args.batch_dir)
    print(f"Analyzing batch: {batch_id}")
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"batch_analysis_{batch_id}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load batch data
    batch_data = load_batch_data(args.batch_dir)
    
    # Extract metrics
    df = extract_metrics(batch_data)
    
    if len(df) == 0:
        print("Error: No valid data found in the batch")
        return
    
    # Calculate detailed metrics by problem type
    metrics = calculate_per_problem_metrics(df)
    
    # Generate visualizations for each problem type
    problem_dirs = {}
    for problem_type in df["problem_type"].unique():
        print(f"Generating visualizations for problem type: {problem_type}")
        problem_dir = generate_problem_visualizations(df, problem_type, args.output_dir)
        problem_dirs[problem_type] = problem_dir
    
    # Generate batch-wide visualizations
    print("Generating batch-wide visualizations")
    generate_batch_visualizations(df, metrics, args.output_dir)
    
    # Generate detailed report
    print("Generating detailed report")
    report_path = generate_detailed_report(metrics, problem_dirs, args.output_dir, batch_id)
    
    # Export data to CSV for further analysis
    df.to_csv(os.path.join(args.output_dir, "batch_data.csv"), index=False)
    
    print(f"Analysis complete! Report available at: {report_path}")
    print(f"All analysis outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
