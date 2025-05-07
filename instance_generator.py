import random
import numpy as np
import json

class CPSATInstanceGenerator:
    """Generator for CP-SAT problem instances with VLLM heuristic integration."""
    
    def __init__(self, seed=None):
        """Initialize the generator with an optional random seed."""
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
    
    def generate_knapsack(self, num_items=50, weight_capacity=1000, 
                         min_value=10, max_value=100, 
                         min_weight=5, max_weight=100):
        """Generate a knapsack problem instance."""
        # Generate item values and weights
        values = [self.rng.randint(min_value, max_value) for _ in range(num_items)]
        weights = [self.rng.randint(min_weight, max_weight) for _ in range(num_items)]
        
        # Create problem description
        problem = {
            "type": "knapsack",
            "num_items": num_items,
            "weight_capacity": weight_capacity,
            "items": [{"id": i, "value": values[i], "weight": weights[i]} 
                      for i in range(num_items)]
        }
        
        # Generate text description
        problem["description"] = self._generate_knapsack_description(problem)
        
        return problem
    
    def _generate_knapsack_description(self, problem):
        """Generate a text description for a knapsack problem."""
        num_items = problem["num_items"]
        capacity = problem["weight_capacity"]
        items = problem["items"]
        
        # Calculate some statistics for the description
        total_value = sum(item["value"] for item in items)
        total_weight = sum(item["weight"] for item in items)
        avg_value = total_value / num_items
        avg_weight = total_weight / num_items
        
        # Sort items by value-to-weight ratio
        items_by_ratio = sorted(items, key=lambda x: x["value"] / x["weight"], reverse=True)
        best_items = items_by_ratio[:min(5, num_items)] # Ensure we don't try to show more items than exist
        
        description = f"""
# Knapsack Problem

## Overview
This is a 0-1 knapsack problem where you need to select items to maximize total value while respecting a weight constraint.

## Problem Details
- Number of items: {num_items}
- Weight capacity: {capacity}
- Total value of all items: {total_value}
"""
        
        if total_weight > capacity:
            description += f"- Total weight of all items: {total_weight} (exceeds capacity by {total_weight - capacity} units)\n"
        elif total_weight == capacity:
            description += f"- Total weight of all items: {total_weight} (equals capacity)\n"
        else:
            description += f"- Total weight of all items: {total_weight} (is {capacity - total_weight} units below capacity)\n"
            
        description += f"""- Average item value: {avg_value:.2f}
- Average item weight: {avg_weight:.2f}

## Top Items by Value-to-Weight Ratio
"""
        
        if not best_items:
            description += "- No items to display.\n"
        else:
            for item in best_items: # Removed enumerate as 'i' was unused and item['id'] is the true identifier
                ratio = item["value"] / item["weight"] if item["weight"] > 0 else float('inf')
                description += f"- Item {item['id']} (variable x_{item['id']}): Value = {item['value']}, Weight = {item['weight']}, Ratio = {ratio:.2f}\n"
        
        description += f"""
## Optimization Goal
Choose a subset of the {num_items} items to include in the knapsack such that:
1. The total weight does not exceed {capacity}
2. The total value is maximized

## Decision Variables
- x_i (binary): 1 if item i is selected, 0 otherwise

## Constraints
- Sum of weights of selected items ≤ {capacity}

## Objective
- Maximize the sum of values of selected items
"""
        
        return description
    
    def generate_scheduling(self, num_jobs=20, num_machines=5, 
                           min_duration=10, max_duration=100,
                           with_precedence=True, precedence_density=0.3):
        """Generate a job scheduling problem instance."""
        # Generate job durations
        durations = [self.rng.randint(min_duration, max_duration) for _ in range(num_jobs)]
        
        # Generate machine eligibility (which machine can process which job)
        eligibility = []
        for j in range(num_jobs):
            eligible_machines = self.rng.sample(
                range(num_machines), 
                self.rng.randint(1, num_machines)
            )
            eligibility.append(eligible_machines)
        
        # Generate precedence constraints if requested
        precedence = []
        if with_precedence:
            # Simple topological ordering to ensure no cycles
            for j1 in range(num_jobs):
                for j2 in range(j1+1, num_jobs):
                    if self.rng.random() < precedence_density:
                        precedence.append((j1, j2))
        
        # Create problem description
        problem = {
            "type": "scheduling",
            "num_jobs": num_jobs,
            "num_machines": num_machines,
            "jobs": [{"id": j, "duration": durations[j], "eligible_machines": eligibility[j]} 
                    for j in range(num_jobs)],
            "precedence": precedence
        }
        
        # Generate text description
        problem["description"] = self._generate_scheduling_description(problem)
        
        return problem
    
    def _generate_scheduling_description(self, problem):
        """Generate a text description for a scheduling problem."""
        num_jobs = problem["num_jobs"]
        num_machines = problem["num_machines"]
        jobs = problem["jobs"]
        precedence = problem["precedence"]
        
        # Calculate some statistics
        total_duration = sum(job["duration"] for job in jobs)
        avg_duration = total_duration / num_jobs
        max_duration = max(job["duration"] for job in jobs)
        min_duration = min(job["duration"] for job in jobs)
        
        # Identify critical jobs
        long_jobs = sorted(jobs, key=lambda x: x["duration"], reverse=True)[:5]
        
        # Count precedence relationships
        prec_count = len(precedence)
        
        description = f"""
# Job Scheduling Problem

## Overview
This is a job scheduling problem where {num_jobs} jobs need to be assigned to {num_machines} machines with timing constraints.

## Problem Details
- Number of jobs: {num_jobs}
- Number of machines: {num_machines}
- Total processing time of all jobs: {total_duration}
- Average job duration: {avg_duration:.2f}
- Job duration range: {min_duration} to {max_duration}
- Number of precedence constraints: {prec_count}

## Job Characteristics
Each job has a specific duration and can only be processed on certain machines:
"""
        
        for i, job in enumerate(long_jobs):
            description += f"- Job {job['id']}: Duration = {job['duration']}, Eligible machines = {job['eligible_machines']}\n"
        
        description += f"\n(Showing 5 longest jobs out of {num_jobs} total jobs)\n"
        
        if precedence:
            description += "\n## Precedence Constraints\n"
            precedence_sample = precedence[:min(5, len(precedence))]
            for j1, j2 in precedence_sample:
                description += f"- Job {j1} must complete before Job {j2} can start\n"
            
            if len(precedence) > 5:
                description += f"(Showing 5 precedence constraints out of {len(precedence)} total)\n"
        
        description += f"""
## Optimization Goal
Assign jobs to machines and determine start times such that:
1. Each job is processed on exactly one eligible machine
2. No machine processes more than one job at a time
3. All precedence constraints are respected
4. The makespan (completion time of the last job) is minimized

## Decision Variables
- start_j: Start time of job j
- end_j: End time of job j
- machine_j: Machine assigned to job j

## Constraints
- Each job is assigned to an eligible machine
- No overlap between jobs on the same machine
- Precedence constraints are respected (if job i must precede job j, then end_i ≤ start_j)
- end_j = start_j + duration_j for all jobs j

## Objective
- Minimize the makespan (maximum completion time across all jobs)
"""
        
        return description
    
    def generate_bin_packing(self, num_items=100, num_bins=20, 
                           bin_capacity=100, min_size=5, max_size=50):
        """Generate a bin packing problem instance."""
        # Generate item sizes
        item_sizes = [self.rng.randint(min_size, max_size) for _ in range(num_items)]
        
        # Create problem description
        problem = {
            "type": "bin_packing",
            "num_items": num_items,
            "num_bins": num_bins,
            "bin_capacity": bin_capacity,
            "items": [{"id": i, "size": item_sizes[i]} for i in range(num_items)]
        }
        
        # Generate text description
        problem["description"] = self._generate_bin_packing_description(problem)
        
        return problem
    
    def _generate_bin_packing_description(self, problem):
        """Generate a text description for a bin packing problem."""
        num_items = problem["num_items"]
        num_bins = problem["num_bins"]
        capacity = problem["bin_capacity"]
        items = problem["items"]
        
        # Calculate some statistics
        total_size = sum(item["size"] for item in items)
        avg_size = total_size / num_items
        max_size = max(item["size"] for item in items)
        min_size = min(item["size"] for item in items)
        
        # Theoretical minimum bins needed
        min_bins_needed = (total_size + capacity - 1) // capacity
        
        # Find largest items
        largest_items = sorted(items, key=lambda x: x["size"], reverse=True)[:5]
        
        description = f"""
# Bin Packing Problem

## Overview
This is a bin packing problem where {num_items} items need to be assigned to at most {num_bins} bins without exceeding bin capacity.

## Problem Details
- Number of items: {num_items}
- Number of available bins: {num_bins}
- Bin capacity: {capacity}
- Total size of all items: {total_size}
- Average item size: {avg_size:.2f}
- Item size range: {min_size} to {max_size}
- Theoretical minimum bins needed: {min_bins_needed}

## Largest Items
"""
        
        for i, item in enumerate(largest_items, 1):
            description += f"- Item {item['id']}: Size = {item['size']} ({item['size']/capacity:.2%} of bin capacity)\n"
        
        description += f"""
## Optimization Goal
Assign each item to exactly one bin such that:
1. The total size of items in each bin does not exceed the bin capacity
2. The number of bins used is minimized

## Decision Variables
- x_i,j (binary): 1 if item i is assigned to bin j, 0 otherwise
- y_j (binary): 1 if bin j is used, 0 otherwise

## Constraints
- Each item is assigned to exactly one bin
- The total size of items in each bin does not exceed the capacity
- Maximum of {num_bins} bins can be used

## Objective
- Minimize the number of bins used
"""
        
        return description
    
    def generate_tsp(self, num_cities=50, min_distance=10, max_distance=1000):
        """Generate a traveling salesman problem instance."""
        # Generate distances between cities
        distances = []
        for i in range(num_cities):
            row = []
            for j in range(num_cities):
                if i == j:
                    row.append(0)
                elif j < i:
                    row.append(distances[j][i])  # Symmetric TSP
                else:
                    row.append(self.rng.randint(min_distance, max_distance))
            distances.append(row)
        
        # Create problem description
        problem = {
            "type": "tsp",
            "num_cities": num_cities,
            "distances": distances
        }
        
        # Generate text description
        problem["description"] = self._generate_tsp_description(problem)
        
        return problem
    
    def _generate_tsp_description(self, problem):
        """Generate a text description for a TSP problem."""
        num_cities = problem["num_cities"]
        distances = problem["distances"]
        
        # Calculate some statistics
        all_distances = []
        for i in range(num_cities):
            for j in range(i+1, num_cities):
                all_distances.append(distances[i][j])
        
        avg_distance = sum(all_distances) / len(all_distances)
        max_distance = max(all_distances)
        min_distance = min(all_distances)
        
        description = f"""
# Traveling Salesman Problem

## Overview
This is a symmetric traveling salesman problem (TSP) where a salesperson must visit each of {num_cities} cities exactly once and return to the starting city, with the goal of minimizing the total travel distance.

## Problem Details
- Number of cities: {num_cities}
- Distance range between cities: {min_distance} to {max_distance}
- Average distance between cities: {avg_distance:.2f}

## Distance Matrix Information
- The distances between cities form a {num_cities}x{num_cities} matrix
- All distances are symmetric (distance from city i to j equals distance from j to i)
- The diagonal contains zeros (distance from a city to itself)

## Optimization Goal
Find a tour (a sequence of cities) such that:
1. Each city is visited exactly once
2. The tour starts and ends at the same city
3. The total travel distance is minimized

## Decision Variables
- x_i,j (binary): 1 if the tour includes a direct trip from city i to city j, 0 otherwise
- u_i (integer): Position of city i in the tour (used to prevent subtours)

## Constraints
- Each city must be entered exactly once
- Each city must be exited exactly once
- No subtours are allowed (the tour must be connected)

## Objective
- Minimize the total distance traveled
"""
        
        return description
    
    def generate_sat(self, num_variables=100, num_clauses=400, clause_size=3):
        """Generate a boolean satisfiability problem instance."""
        # Generate random clauses
        clauses = []
        for _ in range(num_clauses):
            vars_in_clause = self.rng.sample(range(1, num_variables+1), clause_size)
            signs = [self.rng.choice([-1, 1]) for _ in range(clause_size)]
            clause = [(sign * var) for sign, var in zip(signs, vars_in_clause)]
            clauses.append(clause)
        
        # Create problem description
        problem = {
            "type": "sat",
            "num_variables": num_variables,
            "num_clauses": num_clauses,
            "clauses": clauses
        }
        
        # Generate text description
        problem["description"] = self._generate_sat_description(problem)
        
        return problem
    
    def _generate_sat_description(self, problem):
        """Generate a text description for a SAT problem."""
        num_variables = problem["num_variables"]
        num_clauses = problem["num_clauses"]
        clauses = problem["clauses"]
        
        # Calculate statistics
        clause_sizes = [len(clause) for clause in clauses]
        avg_clause_size = sum(clause_sizes) / len(clause_sizes)
        
        # Count variable frequencies
        var_freq = {}
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                var_freq[var] = var_freq.get(var, 0) + 1
        
        most_common_vars = sorted(var_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        description = f"""
# Boolean Satisfiability (SAT) Problem

## Overview
This is a {avg_clause_size:.1f}-SAT problem where the goal is to find a truth assignment for {num_variables} boolean variables that satisfies {num_clauses} clauses.

## Problem Details
- Number of variables: {num_variables}
- Number of clauses: {num_clauses}
- Clause-to-variable ratio: {num_clauses/num_variables:.2f}
- Average clause size: {avg_clause_size:.2f}

## Most Frequently Occurring Variables
"""
        
        for var, freq in most_common_vars:
            description += f"- Variable {var}: appears in {freq} clauses ({freq/num_clauses:.2%} of all clauses)\n"
        
        description += f"""
## Sample Clauses
"""
        
        for i, clause in enumerate(clauses[:5]):
            clause_str = " ∨ ".join([f"x_{abs(lit)}" if lit > 0 else f"¬x_{abs(lit)}" for lit in clause])
            description += f"- Clause {i+1}: ({clause_str})\n"
        
        description += f"(Showing 5 clauses out of {num_clauses} total)\n"
        
        description += f"""
## Optimization Goal
Find an assignment of boolean values to the variables such that:
1. Every clause is satisfied (at least one literal in each clause is true)
2. If multiple satisfying assignments exist, any one will suffice

## Decision Variables
- x_i (boolean): Truth value assigned to variable i

## Constraints
- Each clause must have at least one true literal

## Note
This problem is formulated as a decision problem (find any satisfying assignment) rather than an optimization problem. However, it can be extended to MaxSAT (maximize the number of satisfied clauses) if a fully satisfying assignment doesn't exist.
"""
        
        return description
    
    def generate_sudoku(self, num_filled=17):
        """Generate a Sudoku problem instance (9x9 with some filled cells)."""
        # Create an empty grid
        grid = [[0 for _ in range(9)] for _ in range(9)]
        
        # Function to check if a value can be placed
        def is_valid(row, col, num):
            # Check row and column
            for i in range(9):
                if grid[row][i] == num or grid[i][col] == num:
                    return False
            
            # Check 3x3 box
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    if grid[i][j] == num:
                        return False
            
            return True
        
        # Try to fill cells
        cells = [(i, j) for i in range(9) for j in range(9)]
        self.rng.shuffle(cells)
        
        filled = 0
        for row, col in cells:
            if filled >= num_filled:
                break
                
            if grid[row][col] == 0:
                nums = list(range(1, 10))
                self.rng.shuffle(nums)
                
                for num in nums:
                    if is_valid(row, col, num):
                        grid[row][col] = num
                        filled += 1
                        break
        
        # Create problem description
        problem = {
            "type": "sudoku",
            "grid": grid
        }
        
        # Generate text description
        problem["description"] = self._generate_sudoku_description(problem)
        
        return problem
    
    def _generate_sudoku_description(self, problem):
        """Generate a text description for a Sudoku problem."""
        grid = problem["grid"]
        
        # Count filled cells
        filled_cells = sum(1 for row in grid for cell in row if cell > 0)
        empty_cells = 81 - filled_cells
        
        # Create a visual grid representation
        grid_vis = "```\n"
        for i, row in enumerate(grid):
            if i % 3 == 0 and i > 0:
                grid_vis += "------+-------+------\n"
            
            row_str = ""
            for j, cell in enumerate(row):
                if j % 3 == 0 and j > 0:
                    row_str += "| "
                
                if cell == 0:
                    row_str += ". "
                else:
                    row_str += f"{cell} "
            
            grid_vis += row_str + "\n"
        
        grid_vis += "```"
        
        description = f"""
# Sudoku Puzzle

## Overview
This is a standard 9x9 Sudoku puzzle with {filled_cells} pre-filled cells and {empty_cells} cells to be determined.

## Problem Details
- Board size: 9x9
- Sub-grid size: 3x3
- Pre-filled cells: {filled_cells}
- Empty cells: {empty_cells}

## Current Board State
{grid_vis}

## Optimization Goal
Fill in the empty cells with digits 1-9 such that:
1. Each row contains all digits 1-9 without repetition
2. Each column contains all digits 1-9 without repetition
3. Each 3x3 sub-grid contains all digits 1-9 without repetition
4. The pre-filled cells remain unchanged

## Decision Variables
- x_i,j,d (binary): 1 if cell (i,j) contains digit d, 0 otherwise

## Constraints
- Each cell contains exactly one digit
- Each row contains each digit exactly once
- Each column contains each digit exactly once
- Each 3x3 sub-grid contains each digit exactly once
- Pre-filled cells match their assigned values

## Note
Sudoku is formulated as a constraint satisfaction problem rather than an optimization problem. A valid solution satisfies all constraints.
"""
        
        return description
    
    def generate_custom_problem(self, problem_type, **kwargs):
        """Generate a problem instance based on the specified type."""
        generators = {
            "knapsack": self.generate_knapsack,
            "scheduling": self.generate_scheduling,
            "bin_packing": self.generate_bin_packing,
            "tsp": self.generate_tsp,
            "sat": self.generate_sat,
            "sudoku": self.generate_sudoku
        }
        
        if problem_type not in generators:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        return generators[problem_type](**kwargs)
    
    def save_instance(self, problem, filename):
        """Save a problem instance to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(problem, f, indent=2)
    
    def load_instance(self, filename):
        """Load a problem instance from a JSON file."""
        with open(filename, 'r') as f:
            return json.load(f)