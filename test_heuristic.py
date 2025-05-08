from constraint import Problem
from heuristic import AdvancedHeuristicSolver

def main():
    # 1) Build a simple CSP: vars A,B,C with colors, all different
    problem = Problem(solver=AdvancedHeuristicSolver())
    colors = ["Red", "Green", "Blue"]
    for var in ["A","B","C"]:
        problem.addVariable(var, colors)
    problem.addConstraint(lambda a,b: a!=b, ("A","B"))
    problem.addConstraint(lambda b,c: b!=c, ("B","C"))
    problem.addConstraint(lambda a,c: a!=c, ("A","C"))

    # 2) Solve (will use LLM for var‐selection)
    solutions = problem.getSolutions()
    
    # 3) Print results
    for i, sol in enumerate(solutions, 1):
        print(f"Solution {i}: {sol}")

if __name__ == "__main__":
    main()
