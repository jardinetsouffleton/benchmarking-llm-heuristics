import random
import numpy as np
import networkx as nx
import json

class GraphProblemGenerator:
    """Generator for graph-based problem instances with LLM heuristic integration."""
    
    def __init__(self, seed=None):
        """Initialize the generator with an optional random seed."""
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
    def generate_maxcut(self, num_nodes=50, edge_probability=0.3, min_weight=1, max_weight=10):
        """Generate a Maximum Cut problem instance."""
        # Create a random graph
        G = nx.gnp_random_graph(num_nodes, edge_probability, seed=self.rng.randint(0, 10000))
        
        # Assign random weights to edges
        for u, v in G.edges():
            G[u][v]['weight'] = self.rng.randint(min_weight, max_weight)
        
        # Create problem description
        edges = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
        problem = {
            "type": "maxcut",
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "edges": edges
        }
        
        # Generate text description
        problem["description"] = self._generate_maxcut_description(problem)
        
        return problem
    
    def _generate_maxcut_description(self, problem):
        """Generate a text description for a MaxCut problem."""
        num_nodes = problem["num_nodes"]
        num_edges = problem["num_edges"]
        edges = problem["edges"]
        
        # Calculate statistics
        total_weight = sum(weight for _, _, weight in edges)
        avg_weight = total_weight / num_edges if num_edges > 0 else 0
        max_weight = max([weight for _, _, weight in edges]) if edges else 0
        min_weight = min([weight for _, _, weight in edges]) if edges else 0
        
        # Compute average node degree
        node_degrees = {}
        for u, v, _ in edges:
            node_degrees[u] = node_degrees.get(u, 0) + 1
            node_degrees[v] = node_degrees.get(v, 0) + 1
        avg_degree = sum(node_degrees.values()) / num_nodes
        
        description = f"""
# Maximum Cut Problem

## Overview
This is a Maximum Cut problem where nodes in a graph need to be partitioned into two sets to maximize the weight of edges crossing between the sets.

## Problem Details
- Number of nodes: {num_nodes}
- Number of edges: {num_edges}
- Edge density: {num_edges / (num_nodes * (num_nodes - 1) / 2):.2f}
- Total edge weight: {total_weight}
- Average edge weight: {avg_weight:.2f}
- Weight range: {min_weight} to {max_weight}
- Average node degree: {avg_degree:.2f}

## Sample Edges
"""
        
        for u, v, weight in edges[:5]:
            description += f"- Edge ({u}, {v}): Weight = {weight}\n"
        
        if num_edges > 5:
            description += f"(Showing 5 edges out of {num_edges} total)\n"
        
        description += f"""
## Optimization Goal
Partition the nodes into two sets (S and V-S) such that:
1. Every node is in exactly one set
2. The sum of weights of edges with one endpoint in S and the other in V-S is maximized

## Decision Variables
- x_i (binary): 1 if node i is in set S, 0 if in set V-S

## Objective
- Maximize the sum of weights of edges crossing between the two sets
"""
        
        return description
    
    def generate_mis(self, num_nodes=50, edge_probability=0.3):
        """Generate a Maximum Independent Set problem instance."""
        # Create a random graph
        G = nx.gnp_random_graph(num_nodes, edge_probability, seed=self.rng.randint(0, 10000))
        
        # Create problem description
        edges = list(G.edges())
        problem = {
            "type": "mis",
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "edges": edges
        }
        
        # Generate text description
        problem["description"] = self._generate_mis_description(problem)
        
        return problem
    
    def _generate_mis_description(self, problem):
        """Generate a text description for a MIS problem."""
        num_nodes = problem["num_nodes"]
        num_edges = problem["num_edges"]
        edges = problem["edges"]
        
        # Compute average node degree
        node_degrees = {}
        for u, v in edges:
            node_degrees[u] = node_degrees.get(u, 0) + 1
            node_degrees[v] = node_degrees.get(v, 0) + 1
        avg_degree = sum(node_degrees.values()) / num_nodes if num_nodes > 0 else 0
        
        description = f"""
# Maximum Independent Set Problem

## Overview
This is a Maximum Independent Set problem where the goal is to find the largest subset of nodes in a graph such that no two nodes in the subset are adjacent.

## Problem Details
- Number of nodes: {num_nodes}
- Number of edges: {num_edges}
- Edge density: {num_edges / (num_nodes * (num_nodes - 1) / 2):.2f}
- Average node degree: {avg_degree:.2f}

## Sample Edges
"""
        
        for u, v in edges[:5]:
            description += f"- Edge ({u}, {v})\n"
        
        if num_edges > 5:
            description += f"(Showing 5 edges out of {num_edges} total)\n"
        
        description += f"""
## Optimization Goal
Find a subset S of nodes such that:
1. No two nodes in S are adjacent (connected by an edge)
2. The size of S is maximized

## Decision Variables
- x_i (binary): 1 if node i is in the independent set, 0 otherwise

## Constraints
- For each edge (i,j), at most one of node i or node j can be in the independent set

## Objective
- Maximize the number of nodes in the independent set
"""
        
        return description
    
    def generate_mvc(self, num_nodes=50, edge_probability=0.3):
        """Generate a Minimum Vertex Cover problem instance."""
        # Create a random graph
        G = nx.gnp_random_graph(num_nodes, edge_probability, seed=self.rng.randint(0, 10000))
        
        # Create problem description
        edges = list(G.edges())
        problem = {
            "type": "mvc",
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "edges": edges
        }
        
        # Generate text description
        problem["description"] = self._generate_mvc_description(problem)
        
        return problem
    
    def _generate_mvc_description(self, problem):
        """Generate a text description for a MVC problem."""
        num_nodes = problem["num_nodes"]
        num_edges = problem["num_edges"]
        edges = problem["edges"]
        
        # Compute average node degree
        node_degrees = {}
        for u, v in edges:
            node_degrees[u] = node_degrees.get(u, 0) + 1
            node_degrees[v] = node_degrees.get(v, 0) + 1
        avg_degree = sum(node_degrees.values()) / num_nodes if num_nodes > 0 else 0
        
        description = f"""
# Minimum Vertex Cover Problem

## Overview
This is a Minimum Vertex Cover problem where the goal is to find the smallest subset of nodes such that every edge in the graph is incident to at least one node in the subset.

## Problem Details
- Number of nodes: {num_nodes}
- Number of edges: {num_edges}
- Edge density: {num_edges / (num_nodes * (num_nodes - 1) / 2):.2f}
- Average node degree: {avg_degree:.2f}

## Sample Edges
"""
        
        for u, v in edges[:5]:
            description += f"- Edge ({u}, {v})\n"
        
        if num_edges > 5:
            description += f"(Showing 5 edges out of {num_edges} total)\n"
        
        description += f"""
## Optimization Goal
Find a subset S of nodes such that:
1. Every edge in the graph has at least one endpoint in S
2. The size of S is minimized

## Decision Variables
- x_i (binary): 1 if node i is in the vertex cover, 0 otherwise

## Constraints
- For each edge (i,j), at least one of node i or node j must be in the vertex cover

## Objective
- Minimize the number of nodes in the vertex cover
"""
        
        return description
    
    def generate_graph_coloring(self, num_nodes=50, edge_probability=0.3, max_colors=10):
        """Generate a Graph Coloring problem instance."""
        # Create a random graph
        G = nx.gnp_random_graph(num_nodes, edge_probability, seed=self.rng.randint(0, 10000))
        
        # Create problem description
        edges = list(G.edges())
        problem = {
            "type": "graph_coloring",
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "max_colors": max_colors,
            "edges": edges
        }
        
        # Generate text description
        problem["description"] = self._generate_graph_coloring_description(problem)
        
        return problem
    
    def _generate_graph_coloring_description(self, problem):
        """Generate a text description for a Graph Coloring problem."""
        num_nodes = problem["num_nodes"]
        num_edges = problem["num_edges"]
        max_colors = problem["max_colors"]
        edges = problem["edges"]
        
        # Compute average node degree
        node_degrees = {}
        for u, v in edges:
            node_degrees[u] = node_degrees.get(u, 0) + 1
            node_degrees[v] = node_degrees.get(v, 0) + 1
        avg_degree = sum(node_degrees.values()) / num_nodes if num_nodes > 0 else 0
        
        # Estimate chromatic number lower bound based on cliques
        estimated_chromatic_bound = min(max(node_degrees.values()) + 1 if node_degrees else 1, max_colors)
        
        description = f"""
# Graph Coloring Problem

## Overview
This is a Graph Coloring problem where each node must be assigned a color such that no adjacent nodes have the same color, using the minimum number of colors.

## Problem Details
- Number of nodes: {num_nodes}
- Number of edges: {num_edges}
- Maximum available colors: {max_colors}
- Edge density: {num_edges / (num_nodes * (num_nodes - 1) / 2):.2f}
- Average node degree: {avg_degree:.2f}
- Estimated lower bound on chromatic number: {estimated_chromatic_bound}

## Sample Edges
"""
        
        for u, v in edges[:5]:
            description += f"- Edge ({u}, {v})\n"
        
        if num_edges > 5:
            description += f"(Showing 5 edges out of {num_edges} total)\n"
        
        description += f"""
## Optimization Goal
Assign a color to each node such that:
1. No two adjacent nodes have the same color
2. The number of colors used is minimized

## Decision Variables
- x_i,c (binary): 1 if node i is assigned color c, 0 otherwise
- y_c (binary): 1 if color c is used, 0 otherwise

## Constraints
- Each node must be assigned exactly one color
- Adjacent nodes must have different colors
- Color c can only be used if y_c = 1

## Objective
- Minimize the number of colors used
"""
        
        return description
    
    def save_instance(self, problem, filename):
        """Save a problem instance to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(problem, f, indent=2)
    
    def load_instance(self, filename):
        """Load a problem instance from a JSON file."""
        with open(filename, 'r') as f:
            return json.load(f)
