import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from heuristic import AdvancedHeuristicSolver
from constraint import Problem
from tests.mock_llm_responses import MockResponse


class TestAdvancedHeuristicSolver(unittest.TestCase):
    
    def setUp(self):
        self.solver = AdvancedHeuristicSolver(node_limit=100)
        
        # Create a simple CSP for testing
        self.problem = Problem()
        self.problem.addVariable('A', [1, 2, 3])
        self.problem.addVariable('B', [1, 2, 3])
        # Constraint: A must not equal B
        self.problem.addConstraint(lambda a, b: a != b, ['A', 'B'])
    
    @patch('requests.post')
    def test_variable_selection(self, mock_post):
        # Mock the LLM response for variable selection
        mock_post.return_value = MockResponse({
            "choices": [{"message": {"content": "<answer>A</answer>"}}]
        })
        
        # Test the variable selection method
        unassigned = ['A', 'B']
        domains = {'A': [1, 2, 3], 'B': [1, 2, 3]}
        vconstraints = {'A': [], 'B': []}
        assignments = {}
        
        selected = self.solver.query_next_variable(unassigned, domains, vconstraints, assignments)
        self.assertEqual(selected, 'A')
        
    @patch('requests.post')
    def test_value_ordering(self, mock_post):
        # Mock the LLM response for value ordering
        mock_post.return_value = MockResponse({
            "choices": [{"message": {"content": "<answer>[3, 1, 2]</answer>"}}]
        })
        
        # Test the value ordering method
        variable = 'A'
        values = [1, 2, 3]
        domains = {'A': [1, 2, 3], 'B': [1, 2, 3]}
        vconstraints = {'A': [], 'B': []}
        assignments = {}
        
        ordering = self.solver.query_value_order(variable, values, domains, vconstraints, assignments)
        self.assertEqual(ordering, [3, 1, 2])
    
    @patch('requests.post')
    def test_infeasibility_detection(self, mock_post):
        # Mock the LLM response indicating infeasibility
        mock_post.return_value = MockResponse({
            "choices": [{"message": {"content": "<answer>INFEASIBLE</answer> The problem is unsolvable because..."}}]
        })
        
        # Test infeasibility detection
        unassigned = ['A', 'B']
        domains = {'A': [1, 2, 3], 'B': [1, 2, 3]}
        vconstraints = {'A': [], 'B': []}
        assignments = {}
        
        result = self.solver.query_next_variable(unassigned, domains, vconstraints, assignments)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result[0], "INFEASIBLE")
        self.assertTrue(len(result[1]) > 0)
    
    @patch('requests.post')
    def test_backtrack_suggestion(self, mock_post):
        # Mock the LLM response suggesting backtracking
        mock_post.return_value = MockResponse({
            "choices": [{"message": {"content": "<answer>BACKTRACK</answer>"}}]
        })
        
        # Test backtracking suggestion
        variable = 'A'
        values = [1, 2, 3]
        domains = {'A': [1, 2, 3], 'B': [1, 2, 3]}
        vconstraints = {'A': [], 'B': []}
        assignments = {}
        
        with self.assertRaises(ValueError) as context:
            self.solver.query_value_order(variable, values, domains, vconstraints, assignments)
        self.assertTrue("BACKTRACK" in str(context.exception))
    
    @patch('requests.post')
    def test_full_solution(self, mock_post):
        # Configure the mock to give different responses for different calls
        def side_effect(*args, **kwargs):
            # Parse the request to determine the response
            if isinstance(args[1], dict):
                request_data = args[1]
            else:
                try:
                    request_data = json.loads(args[1])
                except:
                    request_data = kwargs.get('json', {})
                
            user_content = ""
            try:
                user_content = request_data["messages"][1]["content"]
            except (KeyError, IndexError):
                if "messages" in request_data and len(request_data["messages"]) > 1:
                    user_content = request_data["messages"][1].get("content", "")
            
            if "Which of the unassigned variables should be selected next" in user_content:
                # Extract unassigned variables from the content
                if "A" in user_content and "B" in user_content:
                    # First call - select variable A
                    return MockResponse({
                        "choices": [{"message": {"content": "<answer>A</answer>"}}]
                    })
                else:
                    # Second call - select variable B
                    return MockResponse({
                        "choices": [{"message": {"content": "<answer>B</answer>"}}]
                    })
            elif "Return an array with the values in the order" in user_content:
                # Extract the domain values from the content
                domain_start = user_content.find("Domain values for this variable: ")
                if domain_start != -1:
                    domain_end = user_content.find("\n", domain_start)
                    domain_str = user_content[domain_start + len("Domain values for this variable: "):domain_end]
                    domain_values = eval(domain_str)
                    
                    if "variable: A" in user_content:
                        # Sort the domain values in a way that won't cause constraint violation
                        ordered_values = sorted(domain_values)
                        return MockResponse({
                            "choices": [{"message": {"content": f"<answer>{ordered_values}</answer>"}}]
                        })
                    else:
                        # For variable B, make sure not to use the same first value as A
                        ordered_values = sorted(domain_values, reverse=True)
                        return MockResponse({
                            "choices": [{"message": {"content": f"<answer>{ordered_values}</answer>"}}]
                        })
                
            # Default response
            return MockResponse({
                "choices": [{"message": {"content": "<answer>[1, 2, 3]</answer>"}}]
            })
        
        mock_post.side_effect = side_effect
        
        # Solve the CSP problem
        self.problem.setSolver(self.solver)
        solutions = self.problem.getSolutions()
        
        # Verify we got the expected solutions
        self.assertTrue(len(solutions) > 0)
        for solution in solutions:
            self.assertNotEqual(solution['A'], solution['B'])  # Our constraint


if __name__ == '__main__':
    unittest.main()
