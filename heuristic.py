from constraint import Problem, RecursiveBacktrackingSolver
import random

class AdvancedHeuristicSolver(RecursiveBacktrackingSolver):
    def __init__(self, forwardcheck=True):
        super().__init__(forwardcheck)
        # Add custom parameters for your heuristics
        self.random_restarts = 3
        self.tabu_list_size = 10
    
    def getSolutions(self, domains, constraints, vconstraints):
        # Implement a more advanced search approach
        all_solutions = []
        
        # Implement random restarts
        for _ in range(self.random_restarts):
            # Start with random initial assignments
            initial_assignments = self._generate_random_assignment(domains)
            
            # Find solutions from this starting point
            solutions = self.recursiveBacktracking([], domains, vconstraints, initial_assignments, False)
            all_solutions.extend(solutions)
        
        # Remove duplicates
        unique_solutions = []
        seen = set()
        for solution in all_solutions:
            # Convert solution to hashable form
            solution_tuple = tuple(sorted(solution.items()))
            if solution_tuple not in seen:
                seen.add(solution_tuple)
                unique_solutions.append(solution)
        
        return unique_solutions
    
    def _generate_random_assignment(self, domains):
        """Generate a random partial assignment to start the search."""
        assignments = {}
        for variable, domain in list(domains.items())[:2]:  # Assign first 2 variables randomly
            assignments[variable] = random.choice(domain)
        return assignments
    
    def recursiveBacktracking(self, solutions, domains, vconstraints, assignments, single):
        """Customize the recursive backtracking algorithm."""
        # Implement your custom variable selection heuristic
        variable = self._select_next_variable(domains, vconstraints, assignments)
        
        if variable is None:
            # No unassigned variables. We've got a solution.
            solutions.append(assignments.copy())
            return solutions
        
        # Implement your custom value ordering heuristic
        ordered_values = self._order_values(variable, domains, vconstraints, assignments)
        
        assignments[variable] = None
        forwardcheck = self._forwardcheck
        
        if forwardcheck:
            pushdomains = [domains[x] for x in domains if x not in assignments]
        else:
            pushdomains = None
        
        for value in ordered_values:
            assignments[variable] = value
            
            if pushdomains:
                for domain in pushdomains:
                    domain.pushState()
            
            for constraint, variables in vconstraints[variable]:
                if not constraint(variables, domains, assignments, pushdomains):
                    # Value is not good.
                    break
            else:
                # Value is good. Recurse and get next variable.
                self.recursiveBacktracking(solutions, domains, vconstraints, assignments, single)
                if solutions and single:
                    return solutions
            
            if pushdomains:
                for domain in pushdomains:
                    domain.popState()
        
        del assignments[variable]
        return solutions
    
    def _select_next_variable(self, domains, vconstraints, assignments):
        """Advanced variable selection heuristic."""
        unassigned = [var for var in domains if var not in assignments]
        if not unassigned:
            return None
        
        # Example: Weight-based selection combining multiple heuristics
        def score(var):
            # Domain size (smaller is better)
            domain_score = len(domains[var])
            
            # Constraint count (larger is better)
            constraint_score = len(vconstraints.get(var, []))
            
            # Calculate how many values are likely to be ruled out
            impact_score = 0
            for constraint, variables in vconstraints.get(var, []):
                if all(v in assignments or v == var for v in variables):
                    impact_score += 1
            
            # Combine scores (lower is better)
            return domain_score - (0.5 * constraint_score) - (2 * impact_score)
        
        return min(unassigned, key=score)
    
    def _order_values(self, variable, domains, vconstraints, assignments):
        """Advanced value ordering heuristic."""
        values = domains[variable][:]
        
        # Implement the Least Constraining Value (LCV) heuristic
        # This counts how many values would remain available for 
        # neighboring variables after assigning each value
        def count_remaining_values(value):
            # Temporarily assign the value
            assignments[variable] = value
            try:
                count = 0
                
                # Check impact on unassigned neighbors
                for var_neighbor in domains: # Renamed var to var_neighbor
                    if var_neighbor != variable and var_neighbor not in assignments:
                        # Get variables related to var_neighbor through constraints
                        related = False
                        # Check if var_neighbor is constrained with 'variable'
                        for _constraint, _variables_in_constraint in vconstraints.get(var_neighbor, []):
                            if variable in _variables_in_constraint:
                                related = True
                                break
                        
                        if related:
                            # Count valid remaining values in domain of var_neighbor
                            for val_neighbor in domains[var_neighbor]: # Renamed val to val_neighbor
                                assignments[var_neighbor] = val_neighbor # Renamed var to var_neighbor
                                valid = True
                                try:
                                    for constraint_fn, constrained_vars in vconstraints.get(var_neighbor, []): # Renamed var to var_neighbor, constraint to constraint_fn, variables to constrained_vars
                                        if not constraint_fn(constrained_vars, domains, assignments, None):
                                            valid = False
                                            break
                                    
                                    if valid:
                                        count += 1
                                finally:
                                    del assignments[var_neighbor] # Renamed var to var_neighbor, ensured by finally
            finally:
                # Remove the temporary assignment
                del assignments[variable] # Ensured by finally
            return count
        
        # Sort values by the number of remaining values (more is better)
        return sorted(values, key=count_remaining_values, reverse=True)