from constraint import Problem
from heuristic import AdvancedHeuristicSolver # Assuming heuristic.py is in the same directory or accessible via PYTHONPATH

def run_test():
    # 1. Define a simple CSP problem
    # Example: Map coloring problem (Australia)
    # Variables: WA, NT, Q, NSW, V, SA, T (Western Australia, Northern Territory, Queensland, New South Wales, Victoria, South Australia, Tasmania)
    # Domains: Red, Green, Blue
    # Constraints: Adjacent regions must have different colors.

    problem = Problem(AdvancedHeuristicSolver())

    variables = ["WA", "NT", "Q", "NSW", "V", "SA", "T"]
    colors = ["Red", "Green", "Blue"]

    problem.addVariables(variables, colors)

    # Define adjacencies
    adjacencies = [
        ("WA", "NT"), ("WA", "SA"),
        ("NT", "SA"), ("NT", "Q"),
        ("SA", "Q"), ("SA", "NSW"), ("SA", "V"),
        ("Q", "NSW"),
        ("NSW", "V")
        # Tasmania (T) is not adjacent to any mainland state in this simplified model
    ]

    for r1, r2 in adjacencies:
        problem.addConstraint(lambda color1, color2: color1 != color2, (r1, r2))

    print("Solving a map coloring problem using AdvancedHeuristicSolver...")
    
    # 2. Get solutions using your custom solver
    # The solver is passed during Problem initialization, 
    # or can be passed to getSolutions: solutions = problem.getSolutions(solver=AdvancedHeuristicSolver())
    solutions = problem.getSolutions()

    # 3. Print results
    if solutions:
        print(f"Found {len(solutions)} solution(s).")
        print("First solution found:")
        for var in sorted(solutions[0].keys()):
            print(f"  {var}: {solutions[0][var]}")
        # You can print more solutions if needed
        # for i, sol in enumerate(solutions[:3]): # Print first 3 solutions
        #     print(f"\nSolution {i+1}:")
        #     for var in sorted(sol.keys()):
        #         print(f"  {var}: {sol[var]}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    run_test()
