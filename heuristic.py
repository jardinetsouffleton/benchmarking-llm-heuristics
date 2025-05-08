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
            "infeasible_detected": False,
            "infeasibility_explanation": None  # Added field for explanation
        }
        
        # Setup logging
        self.logger = logging.getLogger("CSP_LLM_Solver")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
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
            "infeasible_detected": False,
            "infeasibility_explanation": None  # Reset explanation
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
    
    def recursiveBacktracking(self, solutions, domains, vconstraints, assignments, single):
        """Customize the recursive backtracking algorithm."""
        # Check if we've reached the node limit
        self.nodes_explored += 1
        if self.nodes_explored >= self.node_limit:
            self.logger.warning(f"Node limit reached: {self.node_limit}")
            return solutions
            
        variable = self.select_next_variable(domains, vconstraints, assignments)
        
        # Handle infeasibility detection from LLM
        if isinstance(variable, tuple) and variable[0] == "INFEASIBLE":
            self.logger.info("LLM detected infeasibility, stopping search")
            self.solving_stats["infeasible_detected"] = True
            self.solving_stats["infeasibility_explanation"] = variable[1]  # Store the explanation
            return solutions
            
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
        
        # Handle tuple return for infeasibility case
        if isinstance(llm_choice, tuple) and llm_choice[0] == "INFEASIBLE":
            self.logger.info(f"LLM declared problem infeasible: {llm_choice[1][:100]}...")
            return llm_choice
            
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
                        f"{user_prompt}\n\n"
                        f"Your previous response caused this error:\n{error_msg}\n"
                        "Please try again with a corrected answer."
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

    def query_next_variable(self, unassigned, domains, vconstraints, assignments):
        """Query the local vllm instance for which variable to pick next.
        
        This function asks the LLM to select the next variable to assign from the 
        unassigned variables. It's the variable selection heuristic.
        """
        prompt = (
            f"Unassigned variables: {unassigned}\n"
            f"Domains of unassigned variables: { {var: domains[var] for var in unassigned} }\n"
            f"Current assignments: {assignments}\n"
            f"Constraints on each var: { {k:len(vconstraints.get(k,[])) for k in domains} }\n\n"
            "You are selecting the next variable to be assigned in a constraint satisfaction problem.\n"
            "Which of the unassigned variables should be selected next?\n\n"
            "If you believe this problem is infeasible based on the current assignments, reply with <answer>INFEASIBLE</answer> "
            "and then provide a detailed explanation of why you believe it's infeasible.\n"
            "Otherwise, provide ONLY the variable name between <answer> tags.\n"
            "Example: <answer>A</answer>"
        )
        
        system_prompt = "You are a CSP solver assistant tasked with selecting the next variable."
        
        def parse_and_validate(content):
            var_name = self._extract_tagged_answer(content)
            
            # Check for infeasibility
            if var_name == "INFEASIBLE":
                # Extract the explanation from the content after the INFEASIBLE tag
                full_content = content.strip()
                explanation = full_content.split("</answer>", 1)
                if len(explanation) > 1:
                    explanation_text = explanation[1].strip()
                else:
                    explanation_text = "No explanation provided"
                
                return ("INFEASIBLE", explanation_text)
                
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
        prompt = (
            f"Variable to assign: {variable}\n"
            f"Domain values for this variable: {values}\n"
            f"Domains of all variables: {domains}\n"
            f"Current assignments: {assignments}\n"
            f"Constraints on this variable: {len(vconstraints.get(variable, []))}\n"
            f"Constraints details: {vconstraints.get(variable, [])}\n\n"
            f"You are deciding the order to try values for variable '{variable}'.\n"
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