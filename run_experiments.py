import os
import json
import time
import logging
import argparse
from datetime import datetime
from constraint import Problem, RecursiveBacktrackingSolver, AllDifferentConstraint
from heuristic import AdvancedHeuristicSolver
from graph_instance_generator import GraphProblemGenerator
import concurrent.futures  # Added for parallel execution

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiments.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CSP_Experiments")

def setup_csp_from_graph_problem(problem, node_limit=10000):
    """Convert a graph problem to a CSP."""
    csp = Problem(solver=AdvancedHeuristicSolver(log_level=logging.INFO, node_limit=node_limit))
    
    if problem["type"] == "maxcut":
        # MaxCut: Binary variable for each node
        for node in range(problem["num_nodes"]):
            csp.addVariable(f"node_{node}", [0, 1])  # 0=Set 0, 1=Set 1
        
        # No constraints, but we'll add an objective function
        # (handled externally since python-constraint doesn't support optimization)
        
    elif problem["type"] == "mis":
        # MIS: Binary variable for each node
        for node in range(problem["num_nodes"]):
            csp.addVariable(f"node_{node}", [0, 1])  # 0=Not in set, 1=In set
        
        # Adjacent nodes can't both be in the set
        for u, v in problem["edges"]:
            csp.addConstraint(lambda a, b: not (a == 1 and b == 1), 
                            (f"node_{u}", f"node_{v}"))
            
    elif problem["type"] == "mvc":
        # MVC: Binary variable for each node
        for node in range(problem["num_nodes"]):
            csp.addVariable(f"node_{node}", [0, 1])  # 0=Not in cover, 1=In cover
        
        # Each edge must have at least one endpoint in the cover
        for u, v in problem["edges"]:
            csp.addConstraint(lambda a, b: a == 1 or b == 1, 
                            (f"node_{u}", f"node_{v}"))
            
    elif problem["type"] == "graph_coloring":
        # Graph Coloring: Color variable for each node
        colors = list(range(1, problem["max_colors"] + 1))
        for node in range(problem["num_nodes"]):
            csp.addVariable(f"node_{node}", colors)
        
        # Adjacent nodes must have different colors
        for u, v in problem["edges"]:
            csp.addConstraint(lambda a, b: a != b, 
                            (f"node_{u}", f"node_{v}"))
            
    return csp

def setup_classical_csp_solver(problem, node_limit=10000):
    """Create a classical CP solver with same constraints but default heuristics."""
    # Use a standard backtracking solver with node limit tracking
    class LimitedBacktrackingSolver(RecursiveBacktrackingSolver):
        def __init__(self, node_limit=node_limit):
            super().__init__()
            self.node_limit = node_limit
            self.nodes_explored = 0
            
        def recursiveBacktracking(self, solutions, domains, vconstraints, assignments, single):
            # Check node limit
            self.nodes_explored += 1
            if self.nodes_explored >= self.node_limit:
                return solutions
            
            # Continue with normal backtracking
            return super().recursiveBacktracking(solutions, domains, vconstraints, assignments, single)
    
    # Use our modified solver with node limit
    csp = Problem(solver=LimitedBacktrackingSolver(node_limit=node_limit))
    
    # Apply the same variables and constraints as the LLM version
    if problem["type"] == "maxcut":
        # MaxCut: Binary variable for each node
        for node in range(problem["num_nodes"]):
            csp.addVariable(f"node_{node}", [0, 1])  # 0=Set 0, 1=Set 1
        
    elif problem["type"] == "mis":
        # MIS: Binary variable for each node
        for node in range(problem["num_nodes"]):
            csp.addVariable(f"node_{node}", [0, 1])  # 0=Not in set, 1=In set
        
        # Adjacent nodes can't both be in the set
        for u, v in problem["edges"]:
            csp.addConstraint(lambda a, b: not (a == 1 and b == 1), 
                            (f"node_{u}", f"node_{v}"))
            
    elif problem["type"] == "mvc":
        # MVC: Binary variable for each node
        for node in range(problem["num_nodes"]):
            csp.addVariable(f"node_{node}", [0, 1])  # 0=Not in cover, 1=In cover
        
        # Each edge must have at least one endpoint in the cover
        for u, v in problem["edges"]:
            csp.addConstraint(lambda a, b: a == 1 or b == 1, 
                            (f"node_{u}", f"node_{v}"))
            
    elif problem["type"] == "graph_coloring":
        # Graph Coloring: Color variable for each node
        colors = list(range(1, problem["max_colors"] + 1))
        for node in range(problem["num_nodes"]):
            csp.addVariable(f"node_{node}", colors)
        
        # Adjacent nodes must have different colors
        for u, v in problem["edges"]:
            csp.addConstraint(lambda a, b: a != b, 
                            (f"node_{u}", f"node_{v}"))
            
    return csp

def evaluate_solution(problem, solution):
    """Evaluate the quality of a solution based on the problem type."""
    if not solution:
        return {"score": 0, "valid": False, "message": "No solution found"}
    
    if problem["type"] == "maxcut":
        # Compute cut value
        cut_value = 0
        for u, v, weight in problem["edges"]:
            if solution[f"node_{u}"] != solution[f"node_{v}"]:
                cut_value += weight
        return {"score": cut_value, "valid": True, "message": f"Cut value: {cut_value}"}
        
    elif problem["type"] == "mis":
        # Count nodes in independent set
        nodes_in_set = sum(1 for var, val in solution.items() if val == 1)
        # Verify independence
        for u, v in problem["edges"]:
            if solution[f"node_{u}"] == 1 and solution[f"node_{v}"] == 1:
                return {"score": 0, "valid": False, "message": f"Invalid: Nodes {u} and {v} are adjacent and both in the set"}
        return {"score": nodes_in_set, "valid": True, "message": f"Independent set size: {nodes_in_set}"}
        
    elif problem["type"] == "mvc":
        # Count nodes in vertex cover
        nodes_in_cover = sum(1 for var, val in solution.items() if val == 1)
        # Verify coverage
        for u, v in problem["edges"]:
            if solution[f"node_{u}"] == 0 and solution[f"node_{v}"] == 0:
                return {"score": float('inf'), "valid": False, "message": f"Invalid: Edge ({u},{v}) is not covered"}
        return {"score": nodes_in_cover, "valid": True, "message": f"Vertex cover size: {nodes_in_cover}"}
        
    elif problem["type"] == "graph_coloring":
        # Count colors used
        colors_used = len(set(solution.values()))
        # Verify coloring
        for u, v in problem["edges"]:
            if solution[f"node_{u}"] == solution[f"node_{v}"]:
                return {"score": float('inf'), "valid": False, "message": f"Invalid: Adjacent nodes {u} and {v} have same color {solution[f'node_{u}']}"}
        return {"score": colors_used, "valid": True, "message": f"Number of colors used: {colors_used}"}
    
    return {"score": 0, "valid": False, "message": "Unknown problem type"}

def run_experiment(problem, time_limit=60, node_limit=10000, output_dir="results"):
    """Run an experiment for a given problem."""
    problem_type = problem["type"]
    problem_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{problem_type}_{problem_id}.json")
    
    logger.info(f"Starting experiment for {problem_type} problem with {problem['num_nodes']} nodes")
    
    # ------- LLM-based solver -------
    logger.info("Running LLM-based solver")
    llm_result = run_llm_solver(problem, time_limit, node_limit)
    
    # ------- Classical solver -------
    logger.info("Running classical solver for comparison")
    classical_result = run_classical_solver(problem, time_limit, node_limit)
    
    # Combine results
    result = {
        "problem_type": problem_type,
        "problem_id": problem_id, 
        "num_nodes": problem["num_nodes"],
        "num_edges": problem["num_edges"],
        "llm_solver": llm_result,
        "classical_solver": classical_result,
        "comparative_analysis": compare_solvers(llm_result, classical_result, problem_type)
    }
    
    # Save result
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Experiment completed. Results saved to {result_file}")
    return result

def run_llm_solver(problem, time_limit, node_limit):
    """Run the experiment with LLM-based solver."""
    # Convert to CSP
    csp = setup_csp_from_graph_problem(problem, node_limit=node_limit)
    solver = csp._solver
    
    # Run the solver with time limit
    start_time = time.time()
    try:
        # Set a timeout if supported
        solutions = csp.getSolutions()
        solve_time = time.time() - start_time
        
        # Get solving statistics
        solving_stats = solver.get_solving_stats()
        
        # Evaluate solutions
        best_solution, best_score, best_evaluation = select_best_solution(solutions, problem)
        
        result = {
            "solve_time": solve_time,
            "solutions_found": len(solutions),
            "best_solution": best_solution,
            "best_score": best_score,
            "evaluation": best_evaluation["message"] if best_evaluation else "No valid solutions found",
            "success": best_solution is not None,
            "solving_stats": solving_stats
        }
        
    except Exception as e:
        solve_time = time.time() - start_time
        logger.error(f"Error in LLM solver: {str(e)}")
        # Try to get partial solving stats if available
        solving_stats = {}
        if hasattr(csp, '_solver') and hasattr(csp._solver, 'get_solving_stats'):
            solving_stats = csp._solver.get_solving_stats()
            
        result = {
            "solve_time": solve_time,
            "solutions_found": 0,
            "error": str(e),
            "success": False,
            "solving_stats": solving_stats
        }
    
    return result

def run_classical_solver(problem, time_limit, node_limit=10000):
    """Run the experiment with classical solver."""
    # Convert to CSP
    csp = setup_classical_csp_solver(problem, node_limit=node_limit)
    
    # Run the solver with time limit
    start_time = time.time()
    try:
        # Set a timeout handler to prevent infinite solving
        def timeout_handler():
            if time.time() - start_time > time_limit:
                logger.warning(f"Time limit of {time_limit}s reached for classical solver")
                raise TimeoutError("Solver time limit reached")
            
        # Store original solutions and patch with timeout check
        original_get_solutions = csp.getSolutions
        
        def get_solutions_with_timeout(*args, **kwargs):
            timeout_handler()
            return original_get_solutions(*args, **kwargs)
        
        # Replace with our timed version
        csp.getSolutions = get_solutions_with_timeout
        
        # Run solver
        solutions = csp.getSolutions()
        solve_time = time.time() - start_time
        
        # Evaluate solutions
        best_solution, best_score, best_evaluation = select_best_solution(solutions, problem)
        
        result = {
            "solve_time": solve_time,
            "solutions_found": len(solutions),
            "best_solution": best_solution,
            "best_score": best_score,
            "evaluation": best_evaluation["message"] if best_evaluation else "No valid solutions found",
            "success": best_solution is not None
        }
        
    except TimeoutError as e:
        solve_time = time.time() - start_time
        logger.warning(f"Classical solver timed out after {solve_time:.2f}s")
        result = {
            "solve_time": solve_time,
            "solutions_found": 0,
            "error": "Time limit exceeded",
            "success": False
        }
    except Exception as e:
        solve_time = time.time() - start_time
        logger.error(f"Error in classical solver: {str(e)}")
        result = {
            "solve_time": solve_time,
            "solutions_found": 0,
            "error": str(e),
            "success": False
        }
    
    return result

def select_best_solution(solutions, problem):
    """Select the best solution from a list based on the problem type."""
    problem_type = problem["type"]
    best_solution = None
    best_score = float('-inf') if problem_type in ["maxcut", "mis"] else float('inf')
    best_evaluation = None
    
    for solution in solutions:
        evaluation = evaluate_solution(problem, solution)
        if evaluation["valid"]:
            current_score = evaluation["score"]
            is_better = (
                (problem_type in ["maxcut", "mis"] and current_score > best_score) or 
                (problem_type in ["mvc", "graph_coloring"] and current_score < best_score)
            )
            
            if best_solution is None or is_better:
                best_solution = solution
                best_score = current_score
                best_evaluation = evaluation
    
    return best_solution, best_score, best_evaluation

def compare_solvers(llm_result, classical_result, problem_type):
    """Compare the performance of LLM and classical solvers."""
    
    # Cannot compare if either solver failed
    if not llm_result.get("success") or not classical_result.get("success"):
        return {
            "comparable": False,
            "reason": "One or both solvers failed to find a solution"
        }
    
    llm_score = llm_result.get("best_score", 0)
    classical_score = classical_result.get("best_score", 0)
    
    # For maximization problems (maxcut, mis)
    if problem_type in ["maxcut", "mis"]:
        score_diff = llm_score - classical_score
        score_ratio = llm_score / classical_score if classical_score > 0 else float('inf')
        better_solver = "LLM" if llm_score > classical_score else "Classical" if classical_score > llm_score else "Tie"
    
    # For minimization problems (mvc, graph_coloring)
    else:
        score_diff = classical_score - llm_score
        score_ratio = classical_score / llm_score if llm_score > 0 else float('inf')
        better_solver = "LLM" if llm_score < classical_score else "Classical" if classical_score < llm_score else "Tie"
    
    time_diff = classical_result.get("solve_time", 0) - llm_result.get("solve_time", 0)
    faster_solver = "LLM" if time_diff > 0 else "Classical" if time_diff < 0 else "Tie"
    
    return {
        "comparable": True,
        "score_difference": score_diff,
        "score_ratio": score_ratio,
        "time_difference": time_diff,
        "better_quality": better_solver,
        "faster_solver": faster_solver,
        "llm_solutions_count": llm_result.get("solutions_found", 0),
        "classical_solutions_count": classical_result.get("solutions_found", 0)
    }

def run_single_experiment_wrapper(task_args):
    """
    Wrapper function to run a single experiment instance.
    This function is designed to be called by the ProcessPoolExecutor.
    """
    problem_type, nodes, density, instance_num, problem_dir, \
    time_limit, node_limit, seed, batch_total_instances, current_instance_idx = task_args

    try:
        # Each worker should have its own generator, seeded appropriately for reproducibility
        instance_seed = seed + instance_num if seed is not None else None
        generator = GraphProblemGenerator(seed=instance_seed)
        
        instance_id = f"{problem_type}_{instance_num+1}"

        if problem_type == "maxcut":
            problem = generator.generate_maxcut(num_nodes=nodes, edge_probability=density)
        elif problem_type == "mis":
            problem = generator.generate_mis(num_nodes=nodes, edge_probability=density)
        elif problem_type == "mvc":
            problem = generator.generate_mvc(num_nodes=nodes, edge_probability=density)
        elif problem_type == "graph_coloring":
            problem = generator.generate_graph_coloring(num_nodes=nodes, edge_probability=density)
        else:
            raise ValueError(f"Unknown problem type in worker: {problem_type}")

        instance_file = os.path.join(problem_dir, f"instance_{instance_num+1}.json")
        generator.save_instance(problem, instance_file)
        
        # Run the experiment
        experiment_result = run_experiment(
            problem, 
            time_limit=time_limit, 
            node_limit=node_limit, 
            output_dir=problem_dir
        )
        
        return {
            "status": "success",
            "instance_id": instance_id,
            "problem_type": problem_type,
            "nodes": nodes,
            "num_edges": problem["num_edges"],
            "experiment_result": experiment_result,
            "current_instance_idx": current_instance_idx
        }
    except Exception as e:
        logger.error(f"Error in worker for {problem_type} instance {instance_num+1}: {e}", exc_info=True)
        return {
            "status": "error",
            "instance_id": f"{problem_type}_{instance_num+1}",
            "problem_type": problem_type,
            "error": str(e),
            "current_instance_idx": current_instance_idx
        }

def run_batch_experiments(problem_types=None, nodes=10, density=0.3, instances_per_type=25, 
                        output_dir="results", node_limit=100, time_limit=1800, seed=None,
                        num_workers=1):  # Added num_workers
    """Run a batch of experiments across multiple problem types, potentially in parallel."""
    if problem_types is None:
        problem_types = ["maxcut", "mis", "mvc", "graph_coloring"]
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    config = {
        "timestamp": timestamp, "problem_types": problem_types, "nodes": nodes, "density": density,
        "instances_per_type": instances_per_type, "total_instances": len(problem_types) * instances_per_type,
        "node_limit": node_limit, "time_limit": time_limit, "seed": seed, "num_workers": num_workers
    }
    with open(os.path.join(batch_dir, "batch_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    batch_stats = {
        "total_instances": len(problem_types) * instances_per_type, "completed_instances": 0,
        "successful_instances": 0, 
        "problem_stats": {p: {"count": 0, "success_llm": 0, "success_classical": 0, "errors": 0} for p in problem_types},
        "start_time": time.time(), "results": []
    }

    tasks = []
    overall_instance_idx = 0
    for problem_type in problem_types:
        problem_dir = os.path.join(batch_dir, problem_type)
        os.makedirs(problem_dir, exist_ok=True)
        for i in range(instances_per_type):
            overall_instance_idx += 1
            task_args = (
                problem_type, nodes, density, i, problem_dir,
                time_limit, node_limit, seed, 
                batch_stats["total_instances"], overall_instance_idx
            )
            tasks.append(task_args)

    processed_results = []  # To store results in order for logging if desired
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task_idx = {executor.submit(run_single_experiment_wrapper, task): task[8] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_task_idx):
            original_idx = future_to_task_idx[future]
            try:
                worker_result = future.result()
                worker_result["original_idx"] = original_idx  # Keep track for ordered logging
                processed_results.append(worker_result)
            except Exception as exc:
                logger.error(f"Task (original_idx {original_idx}) generated an exception: {exc}", exc_info=True)
                processed_results.append({
                    "status": "error", 
                    "error": str(exc), 
                    "original_idx": original_idx,
                    "problem_type": "unknown_due_to_future_error" 
                })
    
    processed_results.sort(key=lambda r: r.get("original_idx", float('inf')))

    for worker_result in processed_results:
        batch_stats["completed_instances"] += 1
        problem_type = worker_result.get("problem_type", "unknown")

        if problem_type != "unknown" and problem_type not in batch_stats["problem_stats"]:
             batch_stats["problem_stats"][problem_type] = {"count": 0, "success_llm": 0, "success_classical": 0, "errors": 0}

        if worker_result["status"] == "success":
            experiment_result = worker_result["experiment_result"]
            instance_id = worker_result["instance_id"]
            
            logger.info(f"Completed instance {instance_id} [Overall: {worker_result['current_instance_idx']}/{batch_stats['total_instances']}]")

            if problem_type != "unknown":
                batch_stats["problem_stats"][problem_type]["count"] += 1
                if experiment_result["llm_solver"]["success"]:
                    batch_stats["problem_stats"][problem_type]["success_llm"] += 1
                if experiment_result["classical_solver"]["success"]:
                    batch_stats["problem_stats"][problem_type]["success_classical"] += 1
                if experiment_result["llm_solver"]["success"] or experiment_result["classical_solver"]["success"]:
                    batch_stats["successful_instances"] += 1
            
            result_summary = {
                "instance_id": instance_id, "problem_type": problem_type,
                "num_nodes": worker_result["nodes"], "num_edges": worker_result["num_edges"],
                "llm_success": experiment_result["llm_solver"]["success"],
                "llm_time": experiment_result["llm_solver"]["solve_time"],
                "classical_success": experiment_result["classical_solver"]["success"],
                "classical_time": experiment_result["classical_solver"]["solve_time"],
                "comparative": experiment_result.get("comparative_analysis")
            }
            batch_stats["results"].append(result_summary)
        else:
            instance_id = worker_result.get("instance_id", f"unknown_instance_{worker_result['original_idx']}")
            logger.error(f"Failed instance {instance_id} [Overall: {worker_result.get('current_instance_idx', worker_result['original_idx'])}/{batch_stats['total_instances']}]. Error: {worker_result['error']}")
            if problem_type != "unknown":
                 batch_stats["problem_stats"][problem_type]["count"] += 1
                 batch_stats["problem_stats"][problem_type]["errors"] += 1
            batch_stats["results"].append({
                "instance_id": instance_id, "problem_type": problem_type, "status": "error", "error_message": worker_result['error']
            })

        if batch_stats["completed_instances"] > 0:
            batch_stats["elapsed_time"] = time.time() - batch_stats["start_time"]
            avg_time_per_instance = batch_stats["elapsed_time"] / batch_stats["completed_instances"]
            remaining_instances = batch_stats["total_instances"] - batch_stats["completed_instances"]
            est_remaining_time = avg_time_per_instance * remaining_instances
            logger.info(f"Progress: {batch_stats['completed_instances']}/{batch_stats['total_instances']} instances processed.")
            logger.info(f"Estimated time remaining: {est_remaining_time/60:.1f} minutes.")
            
            with open(os.path.join(batch_dir, "batch_stats.json"), 'w') as f:
                json.dump(batch_stats, f, indent=2)
    
    total_time = time.time() - batch_stats["start_time"]
    avg_time_val = (total_time / batch_stats["total_instances"]) if batch_stats["total_instances"] > 0 else 0
    success_llm_val = (sum(stats["success_llm"] for stats in batch_stats["problem_stats"].values()) / batch_stats["total_instances"]) if batch_stats["total_instances"] > 0 else 0
    success_classical_val = (sum(stats["success_classical"] for stats in batch_stats["problem_stats"].values()) / batch_stats["total_instances"]) if batch_stats["total_instances"] > 0 else 0
    
    summary = {
        "batch_id": timestamp, "total_instances": batch_stats["total_instances"],
        "successful_instances": batch_stats["successful_instances"],
        "total_time_hours": total_time / 3600,
        "average_time_per_instance": avg_time_val,
        "problem_stats": batch_stats["problem_stats"],
        "success_rate_llm": success_llm_val,
        "success_rate_classical": success_classical_val
    }
    
    with open(os.path.join(batch_dir, "batch_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Batch experiments completed. Total time: {total_time/3600:.2f} hours")
    logger.info(f"Results saved to {batch_dir}")
    
    return batch_stats

def main():
    parser = argparse.ArgumentParser(description="Run CSP experiments with LLM heuristics")
    parser.add_argument("--problem", type=str, choices=["maxcut", "mis", "mvc", "graph_coloring", "all"], default="maxcut",
                         help="Problem type to solve (use 'all' for batch mode)")
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes in the graph")
    parser.add_argument("--density", type=float, default=0.3, help="Edge density")
    parser.add_argument("--instances", type=int, default=1, help="Number of instances to generate per problem type")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--node-limit", type=int, default=100, help="Maximum number of nodes to explore in search")
    parser.add_argument("--time-limit", type=int, default=1800, help="Time limit in seconds per instance")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode across all problem types")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for batch mode")  # Added workers
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.batch or args.problem == "all":
        problem_types = ["maxcut", "mis", "mvc", "graph_coloring"]
        run_batch_experiments(
            problem_types=problem_types, nodes=args.nodes, density=args.density,
            instances_per_type=args.instances, output_dir=args.output,
            node_limit=args.node_limit, time_limit=args.time_limit, seed=args.seed,
            num_workers=args.workers  # Pass num_workers
        )
        return
    
    generator = GraphProblemGenerator(seed=args.seed)
    
    for i in range(args.instances):
        logger.info(f"Generating instance {i+1}/{args.instances} of {args.problem}")
        
        if args.problem == "maxcut":
            problem = generator.generate_maxcut(num_nodes=args.nodes, edge_probability=args.density)
        elif args.problem == "mis":
            problem = generator.generate_mis(num_nodes=args.nodes, edge_probability=args.density)
        elif args.problem == "mvc":
            problem = generator.generate_mvc(num_nodes=args.nodes, edge_probability=args.density)
        elif args.problem == "graph_coloring":
            problem = generator.generate_graph_coloring(num_nodes=args.nodes, edge_probability=args.density)
        else:
            logger.error(f"Invalid problem type: {args.problem}")
            return
        
        instance_file = os.path.join(args.output, f"{args.problem}_instance_{i+1}.json")
        os.makedirs(args.output, exist_ok=True)
        generator.save_instance(problem, instance_file)
        
        run_experiment(problem, time_limit=args.time_limit, node_limit=args.node_limit, output_dir=args.output)

if __name__ == "__main__":
    main()
