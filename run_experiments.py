import os
import json
import time
import logging
import argparse
from datetime import datetime
from constraint import Problem, RecursiveBacktrackingSolver, AllDifferentConstraint
from heuristic import AdvancedHeuristicSolver
from graph_instance_generator import GraphProblemGenerator

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

def main():
    parser = argparse.ArgumentParser(description="Run CSP experiments with LLM heuristics")
    parser.add_argument("--problem", type=str, choices=["maxcut", "mis", "mvc", "graph_coloring"], default="maxcut",
                         help="Problem type to solve")
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes in the graph")
    parser.add_argument("--density", type=float, default=0.3, help="Edge density")
    parser.add_argument("--instances", type=int, default=1, help="Number of instances to generate and solve")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--node-limit", type=int, default=100, help="Maximum number of nodes to explore")
    parser.add_argument("--time-limit", type=int, default=1800, help="Time limit in seconds")
    args = parser.parse_args()
    
    # Initialize the generator
    generator = GraphProblemGenerator(seed=args.seed)
    
    # Generate and solve instances
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
        
        # Save the problem instance
        instance_file = os.path.join(args.output, f"{args.problem}_instance_{i+1}.json")
        os.makedirs(args.output, exist_ok=True)
        generator.save_instance(problem, instance_file)
        
        # Run the experiment with both solvers
        run_experiment(problem, time_limit=args.time_limit, node_limit=args.node_limit, output_dir=args.output)

if __name__ == "__main__":
    main()
