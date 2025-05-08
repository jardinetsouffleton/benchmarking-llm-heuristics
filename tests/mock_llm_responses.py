class MockLLMResponse:
    """Utility class to create mock LLM responses for testing."""
    
    @staticmethod
    def variable_selection(variable_name):
        """Generate a mock response for variable selection."""
        return {
            "choices": [{"message": {"content": f"<answer>{variable_name}</answer>"}}]
        }
    
    @staticmethod
    def value_ordering(values_list):
        """Generate a mock response for value ordering."""
        value_str = str(values_list).replace("'", '"')
        return {
            "choices": [{"message": {"content": f"<answer>{value_str}</answer>"}}]
        }
    
    @staticmethod
    def infeasible(explanation="This problem is unsatisfiable due to conflicting constraints"):
        """Generate a mock response indicating infeasibility."""
        return {
            "choices": [{"message": {"content": f"<answer>INFEASIBLE</answer> {explanation}"}}]
        }
    
    @staticmethod
    def backtrack():
        """Generate a mock response suggesting backtracking."""
        return {
            "choices": [{"message": {"content": "<answer>BACKTRACK</answer>"}}]
        }
    
    @staticmethod
    def error_response():
        """Generate a syntactically invalid response to test error handling."""
        return {
            "choices": [{"message": {"content": "This response has no proper tags"}}]
        }


class MockResponse:
    """Mock HTTP response object for testing."""
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        
    def json(self):
        return self.json_data
        
    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"HTTP Error: {self.status_code}")
