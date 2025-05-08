import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from heuristic import AdvancedHeuristicSolver
from constraint import Problem
from tests.mock_llm_responses import MockLLMResponse, MockResponse


class TestGraphProblems(unittest.TestCase):
    
    def setUp(self):
        self.solver = AdvancedHeuristicSolver(node_limit=1000)
    
    def create_graph_coloring_problem(self, graph):
        problem = Problem(AdvancedHeuristicSolver())
        
        # Add variables - one for each node
        for node in graph.nodes():
            problem.addVariable(str(node), [0, 1, 2])  # 3 colors
        
        # Add constraints - adjacent nodes must have different colors
        for u, v in graph.edges():
            problem.addConstraint(lambda x, y: x != y, [str(u), str(v)])
            
        return problem
    
    @patch('requests.post')
    def test_small_graph_coloring(self, mock_post):
        # Create a small graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])
        
        # Configure mock to simulate intelligent variable and value selection
        def side_effect(*args, **kwargs):
            # Parse the request to determine what's being asked
            request_data = kwargs.get('json', {})
            user_content = request_data.get("messages", [{}])[1].get("content", "")
            
            # Variable selection strategy: highest degree first
            if "Unassigned variables" in user_content and "Which of the unassigned" in user_content:
                # Extract unassigned variables
                unassigned_str = user_content.split("Unassigned variables: ")[1].split("\n")[0]
                unassigned = eval(unassigned_str)
                
                # In our small graph, node 3 has the highest degree
                if "3" in unassigned:
                    return MockResponse(MockLLMResponse.variable_selection("3"))
                elif "1" in unassigned:
                    return MockResponse(MockLLMResponse.variable_selection("1"))
                elif "2" in unassigned:
                    return MockResponse(MockLLMResponse.variable_selection("2"))
                else:
                    return MockResponse(MockLLMResponse.variable_selection("4"))
                    
            # Value ordering strategy: match the exact domain values
            elif "Return an array with the values" in user_content:
                # Extract the domain values from the content to ensure we return exactly what's expected
                domain_str = user_content.split("Domain values for this variable: ")[1].split("\n")[0]
                domain_values = eval(domain_str)
                
                # Return the exact domain values in the order we want them tried
                return MockResponse(MockLLMResponse.value_ordering(domain_values))
            
            # Default response
            return MockResponse(MockLLMResponse.value_ordering([0, 1, 2]))
        
        mock_post.side_effect = side_effect
        
        # Set up the graph coloring problem
        problem = self.create_graph_coloring_problem(G)
        
        # Solve it
        solutions = problem.getSolutions()
        
        # Verify the solutions
        self.assertTrue(len(solutions) > 0)
        
        # Verify each solution is a valid coloring
        for solution in solutions:
            for u, v in G.edges():
                self.assertNotEqual(solution[str(u)], solution[str(v)], 
                                   f"Adjacent nodes {u} and {v} have the same color")


if __name__ == '__main__':
    unittest.main()
