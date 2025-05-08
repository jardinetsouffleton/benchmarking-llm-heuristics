from constraint import Problem, RecursiveBacktrackingSolver
import random
import requests
import logging
import re
import json
import time

class AdvancedHeuristicSolver(RecursiveBacktrackingSolver):
    def __init__(self, forwardcheck=True, log_level=logging.INFO, max_llm_retries=3, node_limit=10000):
        super().__init__(forwardcheck)
        # Add custom parameters for your heuristics
        self.random_restarts = 3
        self.tabu_list_size = 10
        
        # LLM configuration
        self.llm_url = "http://localhost:8000/v1/chat/completions"
        self.llm_model = "deepseek-ai/DeepSeek-Prover-V2-7B"
        self.llm_temperature = 0.3
        self.max_llm_retries = max_llm_retries
        
        # Store problem instance information
        self.problem_instance = None
        self.constraint_descriptions = None
        
        # Performance tracking and limits
        self.node_limit = node_limit
        self.nodes_explored = 0
        self.backtrack_count = 0
        self.llm_call_times = []
        self.solving_stats = {
            "nodes_explored": 0,
            "backtrack_count": 0, 
            "llm_calls": 0,
            "llm_call_times": [],
            "variable_selections": [],
            "value_orderings": [],
            "fixpoint_iterations": 0,
        }
        
        # Setup logging
        self.logger = logging.getLogger("CSP_LLM_Solver")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def set_problem_instance(self, problem_instance):
        """Set the complete problem instance data.
        
        Args:
            problem_instance: A dictionary containing all problem information
                              including problem type, description, constraints, etc.
        """
        self.problem_instance = problem_instance
        self.logger.info(f"Set problem instance of type: {problem_instance.get('type', 'unknown')}")
        
        # Extract constraint descriptions from problem instance if available
        if 'description' in problem_instance:
            self.logger.info("Problem instance includes detailed description")
    
    def set_constraint_descriptions(self, constraint_descriptions):
        """Set human-readable descriptions for constraints.
        
        Args:
            constraint_descriptions: A dictionary mapping constraints to human-readable descriptions.
                                     Format: {(var1, var2): "description"} or {vars_tuple: "description"}
        """
        self.constraint_descriptions = constraint_descriptions
        self.logger.info("Set constraint descriptions for better LLM understanding")

    def getSolutions(self, domains, constraints, vconstraints):
        # Reset stats for this solving session
        self.nodes_explored = 0
        self.backtrack_count = 0
        self.llm_call_times = []
        self.solving_stats = {
            "nodes_explored": 0,
            "backtrack_count": 0,
            "llm_calls": 0,
            "llm_call_times": [],
            "variable_selections": [],
            "value_orderings": [],
            "fixpoint_iterations": 0,
        }
        
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
        
        # Update final stats
        self.solving_stats["nodes_explored"] = self.nodes_explored
        self.solving_stats["backtrack_count"] = self.backtrack_count
        self.solving_stats["llm_call_times"] = self.llm_call_times
        
        return unique_solutions
    
    def _generate_random_assignment(self, domains):
        """Generate a random partial assignment to start the search."""
        assignments = {}
        for variable, domain in list(domains.items())[:2]:  # Assign first 2 variables randomly
            assignments[variable] = random.choice(domain)
        return assignments
    
    def _enforce_fixpoint(self, domains, vconstraints, assignments):
        """Prune domains until no further values can be removed."""
        iterations = 0
        changed = True
        while changed:
            changed = False
            for var, domain in domains.items():
                if var in assignments:
                    continue
                to_remove = []
                for val in list(domain):
                    assignments[var] = val
                    for constraint, vars in vconstraints.get(var, []):
                        if not constraint(vars, domains, assignments, None):
                            to_remove.append(val)
                            break
                    assignments.pop(var, None)
                if to_remove:
                    for v in to_remove:
                        domain.remove(v)
                    changed = True
            iterations += 1
        self.solving_stats["fixpoint_iterations"] += iterations

    def recursiveBacktracking(self, solutions, domains, vconstraints, assignments, single):
        """Customize the recursive backtracking algorithm."""
        # Check if we've reached the node limit
        self.nodes_explored += 1
        if self.nodes_explored >= self.node_limit:
            self.logger.warning(f"Node limit reached: {self.node_limit}")
            return solutions
            
        # Apply fixpoint-based domain pruning before picking next variable
        self._enforce_fixpoint(domains, vconstraints, assignments)
        variable = self.select_next_variable(domains, vconstraints, assignments)
        
        if variable is None:
            # No unassigned variables. We've got a solution.
            solutions.append(assignments.copy())
            return solutions
        
        # Get value ordering from LLM
        try:
            ordered_values = self._order_values(variable, domains, vconstraints, assignments)
        except Exception as e:
            self.logger.error(f"Value ordering failed: {e}")
            # Handle the case where LLM suggests backtracking
            if "BACKTRACK" in str(e):
                self.backtrack_count += 1
                self.logger.info(f"Backtracking from variable {variable} as suggested by LLM")
                return solutions
            raise
        
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
    
    def select_next_variable(self, domains, vconstraints, assignments):
        """Advanced variable selection heuristic, now using LLM."""
        unassigned = [var for var in domains if var not in assignments]
        if not unassigned:
            return None

        # Try LLM-based selection, no fallback
        self.logger.info(f"Querying LLM for next variable among {unassigned}")
        llm_choice = self.query_next_variable(unassigned, domains, vconstraints, assignments)
        
        if llm_choice in unassigned:
            self.logger.info(f"LLM selected variable: {llm_choice}")
            return llm_choice
        else:
            self.logger.error(f"LLM returned invalid variable: {llm_choice}")
            raise ValueError(f"LLM returned invalid variable: {llm_choice}")

    def _call_llm(self, system_prompt, user_prompt):
        """Generic LLM call via vllm."""
        self.logger.debug(f"Calling LLM with system: {system_prompt[:50]}...")
        self.logger.debug(f"User prompt: {user_prompt[:100]}...")
        
        start_time = time.time()
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.llm_temperature,
        }
        try:
            resp = requests.post(self.llm_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            elapsed_time = time.time() - start_time
            
            # Track LLM call time
            self.llm_call_times.append(elapsed_time)
            self.solving_stats["llm_calls"] += 1
            
            self.logger.debug(f"LLM call successful in {elapsed_time:.3f}s")
            return data
        except Exception as e:
            self.logger.error(f"LLM API call failed: {str(e)}")
            raise

    def _call_llm_with_retry(self, system_prompt, user_prompt, parser_func, error_msg=""):
        """Call LLM with retry mechanism for parsing errors.
        
        Args:
            system_prompt: The system prompt to send
            user_prompt: The user prompt to send
            parser_func: Function to parse the LLM response (may raise exceptions)
            error_msg: Optional error message from previous attempt
            
        Returns:
            The parsed result
        """
        attempts = 0
        last_exception = None
        
        while attempts < self.max_llm_retries:
            try:
                # Add error context from previous attempt if any
                current_prompt = user_prompt
                if error_msg and attempts > 0:
                    current_prompt = (
                        
                        f"Your previous response caused this error:\n{error_msg}\n"
                        "Please try again with a corrected answer." +
                        f"{user_prompt}\n\n"
                    )
                
                # Call LLM
                data = self._call_llm(system_prompt, current_prompt)
                content = data["choices"][0]["message"]["content"]
                
                # Try to parse the response
                result = parser_func(content)
                return result
                
            except Exception as e:
                attempts += 1
                last_exception = e
                error_msg = str(e)
                self.logger.warning(f"Attempt {attempts}/{self.max_llm_retries} failed: {error_msg}")
                
        # If we get here, all retries failed
        if last_exception:
            raise last_exception

    def _extract_tagged_answer(self, text):
        """Extract content between <answer> tags."""
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        self.logger.warning(f"No tagged answer found in: {text[:100]}...")
        return text.strip()  # Return whole text if no tags found

    def query_next_variable(self, unassigned, domains, vconstraints, assignments, constraint_descriptions=None):
        """Query the local vllm instance for which variable to pick next.
        
        This function asks the LLM to select the next variable to assign from the 
        unassigned variables. It's the variable selection heuristic.
        
        Args:
            unassigned: List of unassigned variables
            domains: Dictionary of domains for each variable
            vconstraints: Dictionary of constraints for each variable
            assignments: Current variable assignments
            constraint_descriptions: Optional dictionary mapping constraint to readable description
                                     Format: {(var1, var2): "description"} or {vars_tuple: "description"}
        """
        # If explicit constraint_descriptions are provided, use them; 
        # otherwise fall back to stored descriptions
        descriptions_to_use = constraint_descriptions or self.constraint_descriptions
        
        # Build constraint information, using descriptions if provided
        constraint_info = []
        
        if descriptions_to_use:
            # Use the provided human-readable constraint descriptions
            for vars_tuple, description in descriptions_to_use.items():
                if any(v in unassigned for v in vars_tuple):
                    constraint_info.append(f"{' and '.join(vars_tuple)}: {description}")
        else:
            # Fall back to generic constraint representation
            for var, clist in vconstraints.items():
                for _, vars in clist:
                    if isinstance(vars, (list, tuple)) and len(vars) >= 2:
                        constraint_info.append(f"Constraint between {' and '.join(vars)}")
        
        # Add constraints that affect unassigned variables
        relevant_constraints = [c for c in constraint_info if any(v in c for v in unassigned)]
        constraint_str = "\n- ".join([""] + sorted(set(relevant_constraints))) if relevant_constraints else "None"

        # Add objective function information based on problem type
        objective_str = ""
        if self.problem_instance:
            problem_type = self.problem_instance.get('type', 'general')
            if problem_type == "maxcut":
                objective_str = "Objective: MAXIMIZE the sum of weights of edges crossing the cut (edges with endpoints in different partitions)."
            elif problem_type == "mis":
                objective_str = "Objective: MAXIMIZE the number of nodes in the independent set."
            elif problem_type == "mvc":
                objective_str = "Objective: MINIMIZE the number of nodes in the vertex cover."
            elif problem_type == "graph_coloring":
                objective_str = "Objective: MINIMIZE the number of colors used while ensuring adjacent nodes have different colors."

        # Build the prompt, including problem instance information if available
        prompt = (
            f"Unassigned variables: {unassigned}\n"
            f"Domains of unassigned variables: {domains}\n"
            f"Current assignments: {assignments}\n"
            f"Constraints:{constraint_str}\n"
        )
        
        # Add problem type and description if available
        if self.problem_instance:
            problem_type = self.problem_instance.get('type', 'general')
            prompt += f"\nProblem type: {problem_type}\n"
            if objective_str:
                prompt += f"{objective_str}\n"
            
            # Add relevant part of problem description if available
            if 'description' in self.problem_instance:
                # Extract just the overview and problem details sections to keep prompt manageable
                desc = self.problem_instance['description']
                overview_match = re.search(r'## Overview(.*?)##', desc, re.DOTALL)
                details_match = re.search(r'## Problem Details(.*?)##', desc, re.DOTALL)
                
                if overview_match:
                    prompt += f"Problem overview: {overview_match.group(1).strip()}\n"
                if details_match:
                    prompt += f"Problem details: {details_match.group(1).strip()}\n"
        
        prompt += (
            "\nYou are selecting the next variable to be assigned in a constraint satisfaction problem.\n"
            "Which of the unassigned variables should be selected next?\n\n"
            "Provide ONLY the variable name between <answer> tags.\n"
            "Example: <answer>A</answer>"
        )
        
        system_prompt = "You are a CSP solver assistant tasked with selecting the next variable."
        
        def parse_and_validate(content):
            var_name = self._extract_tagged_answer(content)
            
            if var_name not in unassigned:
                raise ValueError(f"Variable '{var_name}' is not in the unassigned list: {unassigned}")
                
            # Record selection
            self.solving_stats["variable_selections"].append({
                "variable": var_name, 
                "unassigned": unassigned.copy(),
                "assignments": assignments.copy()
            })
            
            return var_name
            
        return self._call_llm_with_retry(system_prompt, prompt, parse_and_validate)

    def query_value_order(self, variable, values, domains, vconstraints, assignments):
        """Query the local vllm instance for an ordering of values for the selected variable.
        
        This function asks the LLM to order the domain values for the specific variable
        that was selected by the previous call to select_next_variable. The result is
        used to determine the order in which values are tried for this variable only.
        """
        # Add objective function information based on problem type
        objective_str = ""
        if self.problem_instance:
            problem_type = self.problem_instance.get('type', 'general')
            if problem_type == "maxcut":
                objective_str = "Objective: MAXIMIZE the sum of weights of edges crossing the cut (edges with endpoints in different partitions)."
            elif problem_type == "mis":
                objective_str = "Objective: MAXIMIZE the number of nodes in the independent set."
            elif problem_type == "mvc":
                objective_str = "Objective: MINIMIZE the number of nodes in the vertex cover."
            elif problem_type == "graph_coloring":
                objective_str = "Objective: MINIMIZE the number of colors used while ensuring adjacent nodes have different colors."

        # Build constraint information, including all constraints
        constraint_str = ""
        if self.constraint_descriptions:
            # Include all constraints, not just those relevant to the current variable
            all_constraints = []
            for vars_tuple, description in self.constraint_descriptions.items():
                all_constraints.append(f"{' and '.join(vars_tuple)}: {description}")
            
            if all_constraints:
                constraint_str = "\n- ".join(["All constraints:"] + all_constraints)
            else:
                constraint_str = "No constraints specified."
        else:
            # If no constraint descriptions, use generic representation from vconstraints
            all_constraints = []
            for var, clist in vconstraints.items():
                for _, vars in clist:
                    if isinstance(vars, (list, tuple)) and len(vars) >= 2:
                        constraint_desc = f"Constraint between {' and '.join(vars)}"
                        if constraint_desc not in all_constraints:
                            all_constraints.append(constraint_desc)
            
            if all_constraints:
                constraint_str = "\n- ".join(["All constraints:"] + all_constraints)
            else:
                constraint_str = "No constraints identified."

        # Highlight which constraints directly involve the current variable
        if self.constraint_descriptions:
            relevant_constraints = []
            for vars_tuple, description in self.constraint_descriptions.items():
                if variable in vars_tuple:
                    relevant_constraints.append(f"{' and '.join(vars_tuple)}: {description}")
            
            if relevant_constraints:
                constraint_str += "\n\nConstraints involving the current variable:\n- "
                constraint_str += "\n- ".join(relevant_constraints)

        prompt = (
            f"Variable to assign: {variable}\n"
            f"Domain values for this variable: {values}\n"
            f"Domains of all variables: {domains}\n"
            f"Current assignments: {assignments}\n"
            f"{constraint_str}\n"
        )
        
        # Add problem type and objective
        if self.problem_instance:
            problem_type = self.problem_instance.get('type', 'general')
            prompt += f"\nProblem type: {problem_type}\n"
            if objective_str:
                prompt += f"{objective_str}\n"
        
        prompt += (
            f"\nYou are deciding the order to try values for variable '{variable}'.\n"
            "Return an array with the values in the order you recommend trying them.\n\n"
            "If you believe we should backtrack from this variable because no value will lead to a solution, "
            "reply with <answer>BACKTRACK</answer> instead.\n\n"
            "Otherwise, please provide ONLY the array between <answer> tags.\n"
            "Example: <answer>[1, 2, 3]</answer>"
        )
        
        system_prompt = "You are a CSP solver assistant tasked with ordering domain values."
        
        def parse_and_validate(content):
            json_str = self._extract_tagged_answer(content)
            
            # Check for backtrack signal
            if json_str == "BACKTRACK":
                raise ValueError("BACKTRACK")
                
            ordered_values = json.loads(json_str)  # May raise json.JSONDecodeError
            
            # Validate that it contains the same elements
            if set(ordered_values) != set(values):
                missing = set(values) - set(ordered_values)
                extra = set(ordered_values) - set(values)
                error_msg = ""
                if missing:
                    error_msg += f"Missing values: {missing}. "
                if extra:
                    error_msg += f"Extra values: {extra}. "
                raise ValueError(f"Invalid ordering: {error_msg}")
            
            # Record ordering
            self.solving_stats["value_orderings"].append({
                "variable": variable,
                "values": values.copy(),
                "ordering": ordered_values.copy()
            })
                
            return ordered_values
            
        return self._call_llm_with_retry(system_prompt, prompt, parse_and_validate)

    def _order_values(self, variable, domains, vconstraints, assignments):
        """Advanced value ordering heuristic for the selected variable.
        
        This function orders the possible domain values for the specific variable
        that we're currently trying to assign. The ordering determines which value 
        we try first, second, etc. for this specific variable.
        """
        values = domains[variable][:]
        
        # Use LLM-based ordering for this variable's domain values
        self.logger.info(f"Querying LLM for value ordering of variable '{variable}' with domain {values}")
        llm_order = self.query_value_order(variable, values, domains, vconstraints, assignments)
        
        # Ensure it's a permutation of the original domain
        if set(llm_order) == set(values):
            self.logger.info(f"LLM value ordering: {llm_order}")
            return llm_order
        else:
            self.logger.error(f"LLM returned invalid ordering: {llm_order}")
            raise ValueError(f"LLM returned invalid ordering. Got {llm_order}, expected permutation of {values}")

    def get_solving_stats(self):
        """Return solving statistics."""
        return self.solving_stats