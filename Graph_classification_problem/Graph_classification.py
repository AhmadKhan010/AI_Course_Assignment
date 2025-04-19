import networkx as nx
from collections import defaultdict, deque
import itertools
import time

class GraphClassifier:
    def __init__(self, file_path):
        """
        Initialize the GraphClassifier with a file path.
        
        Args:
            file_path (str): Path to the text file containing graph data.
        """
        self.graph = self._read_graph_from_file(file_path)
        self.subgraphs = {
            'cliques': [],
            'chains': [],
            'stars': [],
            'cycles': []
        }
    
    def _read_graph_from_file(self, file_path):
        """
        Read graph data from a text file and create a NetworkX graph.
        
        Args:
            file_path (str): Path to the text file containing graph data.
            
        Returns:
            nx.Graph: A NetworkX graph object.
        """
        G = nx.DiGraph()  # Create a directed graph
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Strip whitespace and split by comma
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        try:
                            source = int(parts[0])
                            target = int(parts[1])
                            
                            # Add weight if available
                            if len(parts) >= 3:
                                weight = float(parts[2])
                                G.add_edge(source, target, weight=weight)
                            else:
                                G.add_edge(source, target)
                        except ValueError:
                            # Skip lines that can't be converted to integers
                            continue
            
            print(f"Graph loaded successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            return G
        except Exception as e:
            print(f"Error reading graph file: {e}")
            return nx.DiGraph()
    
    def find_all_subgraphs(self, min_size=3, max_size=8):
        """
        Find all subgraphs in the graph that match specific patterns.
        
        Args:
            min_size (int): Minimum size of subgraphs to consider.
            max_size (int): Maximum size of subgraphs to consider.
        """
        print("Finding cliques...")
        start_time = time.time()
        self.find_cliques(min_size, max_size)
        print(f"Clique detection time: {time.time() - start_time:.2f} seconds")
        
        print("Finding chains...")
        start_time = time.time()
        self.find_chains(min_size, max_size)
        print(f"Chain detection time: {time.time() - start_time:.2f} seconds")
        
        print("Finding stars...")
        start_time = time.time()
        self.find_stars(min_size, max_size)
        print(f"Star detection time: {time.time() - start_time:.2f} seconds")
        
        print("Finding cycles...")
        start_time = time.time()
        self.find_cycles(min_size, max_size)
        print(f"Cycle detection time: {time.time() - start_time:.2f} seconds")
    
    def find_cliques(self, min_size=3, max_size=8):
        """
        Find all cliques in the graph using NetworkX's built-in algorithm.
        
        A clique is a fully connected subgraph where every node is connected to every other node.
        
        Args:
            min_size (int): Minimum size of cliques to find.
            max_size (int): Maximum size of cliques to find.
        """
        # Create an undirected version of the graph to find cliques
        undirected_graph = self.graph.to_undirected()
        
        # Find all cliques of specified sizes
        cliques = [c for c in nx.find_cliques(undirected_graph) 
                  if min_size <= len(c) <= max_size]
        
        # Sort cliques by size in descending order (larger cliques have higher priority)
        cliques.sort(key=len, reverse=True)
        
        self.subgraphs['cliques'] = cliques
        print(f"Found {len(cliques)} cliques.")
    
    def find_chains(self, min_size=3, max_size=8):
        """
        Find chains in the graph using a more efficient method.
        
        A chain is a linear structure where each internal node connects to exactly two other nodes.
        
        Args:
            min_size (int): Minimum size of chains to find.
            max_size (int): Maximum size of chains to find.
        """
        undirected_graph = self.graph.to_undirected()
        chains = []
        chain_set = set()  # To track unique chains
        
        # For each node, find potential chains starting from it
        for start_node in undirected_graph.nodes():
            # Skip high-degree nodes as they're less likely to be part of chains
            if undirected_graph.degree(start_node) > 5:
                continue
                
            # Use a modified BFS to find chains efficiently
            visited = {node: False for node in undirected_graph.nodes()}
            
            # Start a BFS from this node
            queue = deque([(start_node, [start_node])])
            visited[start_node] = True
            
            while queue and len(chains) < 1000:  # Limit total chains found
                current, path = queue.popleft()
                
                # If path is already at max size, don't expand further
                if len(path) >= max_size:
                    continue
                    
                # Get neighbors that could be part of a chain
                neighbors = [n for n in undirected_graph.neighbors(current) 
                            if not visited[n] and undirected_graph.degree(n) <= 5]
                
                # If this could be the end of a valid chain
                if len(path) >= min_size and len(neighbors) == 0:
                    # Check if this is a valid chain - all internal nodes have degree 2
                    valid = True
                    for i in range(1, len(path) - 1):
                        if undirected_graph.degree(path[i]) != 2:
                            valid = False
                            break
                            
                    if valid:
                        # Check if the start and end nodes have exactly one connection in the chain
                        if (undirected_graph.degree(path[0]) == 1 or sum(1 for n in path[1:] if undirected_graph.has_edge(path[0], n)) == 1) and \
                           (undirected_graph.degree(path[-1]) == 1 or sum(1 for n in path[:-1] if undirected_graph.has_edge(path[-1], n)) == 1):
                            
                            # Create a canonical representation of the chain
                            canonical = tuple(path) if path[0] < path[-1] else tuple(reversed(path))
                            
                            if canonical not in chain_set:
                                chain_set.add(canonical)
                                chains.append(list(canonical))
                
                # Continue the search - only if we haven't found too many chains yet
                if len(chains) < 1000:
                    for neighbor in neighbors:
                        visited[neighbor] = True
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
                        
                        # To avoid memory issues, limit the BFS queue size
                        if len(queue) > 10000:
                            break
        
        # Sort chains by size in descending order
        chains.sort(key=len, reverse=True)
        
        self.subgraphs['chains'] = chains
        print(f"Found {len(chains)} chains.")
    
    def find_stars(self, min_size=3, max_size=8):
        """
        Find all star configurations in the graph.
        
        A star is a central node connected to multiple outer nodes with no connections among the outer nodes.
        
        Args:
            min_size (int): Minimum size of stars to find (including center node).
            max_size (int): Maximum size of stars to find (including center node).
        """
        undirected_graph = self.graph.to_undirected()
        stars = []
        
        # Find nodes with degree that could form stars of the required size
        for node in undirected_graph.nodes():
            neighbors = list(undirected_graph.neighbors(node))
            
            # Skip nodes with too few or too many neighbors
            if len(neighbors) < min_size - 1 or len(neighbors) > max_size - 1:
                continue
            
            # Check if neighbors are not connected to each other
            is_star = True
            
            # Build a set of all neighbors for faster lookup
            neighbor_set = set(neighbors)
            
            for neighbor in neighbors:
                # Check if this neighbor connects to any other neighbor
                for second_neighbor in undirected_graph.neighbors(neighbor):
                    if second_neighbor in neighbor_set and second_neighbor != neighbor:
                        is_star = False
                        break
                
                if not is_star:
                    break
            
            if is_star:
                star = [node] + neighbors
                stars.append(star)
        
        # Sort stars by size in descending order
        stars.sort(key=len, reverse=True)
        
        self.subgraphs['stars'] = stars
        print(f"Found {len(stars)} stars.")
    
    def find_cycles(self, min_size=3, max_size=8):
        """
        Find cycles in the graph using a more efficient approach.
        
        A cycle is a closed path where each node is part of a loop.
        
        Args:
            min_size (int): Minimum size of cycles to find.
            max_size (int): Maximum size of cycles to find.
        """
        cycles = []
        cycle_set = set()
        
        # Use NetworkX's cycle_basis which is efficient for undirected graphs
        undirected_graph = self.graph.to_undirected()
        
        try:
            # Get the cycles from cycle_basis
            for cycle in nx.cycle_basis(undirected_graph):
                if min_size <= len(cycle) <= max_size:
                    # Canonicalize the cycle to avoid duplicates
                    min_val = min(cycle)
                    min_idx = cycle.index(min_val)
                    canonical_cycle = tuple(cycle[min_idx:] + cycle[:min_idx])
                    
                    if canonical_cycle not in cycle_set:
                        cycle_set.add(canonical_cycle)
                        cycles.append(list(canonical_cycle))
        except Exception as e:
            print(f"Error finding cycles: {e}")
            
            # Fallback method - only for smaller graphs or if the main method fails
            if undirected_graph.number_of_nodes() < 100:
                print("Falling back to basic cycle detection...")
                
                for size in range(min_size, min(max_size + 1, 6)):  # Limit to smaller cycles
                    for nodes in itertools.combinations(undirected_graph.nodes(), size):
                        subgraph = undirected_graph.subgraph(nodes)
                        # Check if every node has degree 2 in the subgraph
                        if all(subgraph.degree(node) == 2 for node in subgraph.nodes()):
                            cycle = list(nodes)
                            # Canonicalize
                            min_val = min(cycle)
                            min_idx = cycle.index(min_val)
                            canonical_cycle = tuple(cycle[min_idx:] + cycle[:min_idx])
                            
                            if canonical_cycle not in cycle_set:
                                cycle_set.add(canonical_cycle)
                                cycles.append(list(canonical_cycle))
        
        # Sort cycles by size in descending order
        cycles.sort(key=len, reverse=True)
        
        self.subgraphs['cycles'] = cycles
        print(f"Found {len(cycles)} cycles.")
    
    def print_subgraph_summary(self, show_all=True):
        """
        Print a summary of all found subgraphs.
        
        Args:
            show_all (bool): If True, show all subgraphs. If False, show only top 5.
        """
        print("\n=== Subgraph Classification Summary ===")
        
        for subgraph_type, subgraphs in self.subgraphs.items():
            print(f"\n{subgraph_type.upper()} ({len(subgraphs)} found):")
            
            if show_all or len(subgraphs) <= 10:
                # Show all subgraphs
                for i, subgraph in enumerate(subgraphs):
                    print(f"  {i+1}. Size {len(subgraph)}: {subgraph}")
            else:
                # Show only top 10
                for i, subgraph in enumerate(subgraphs[:10]):
                    print(f"  {i+1}. Size {len(subgraph)}: {subgraph}")
                print(f"  ... and {len(subgraphs) - 10} more {subgraph_type}.")

def main():
    file_path = 'data1.txt'
    
    # Create the classifier
    classifier = GraphClassifier(file_path)
    
    # Find all subgraphs with a reasonable size constraint
    start_time = time.time()
    classifier.find_all_subgraphs(min_size=3, max_size=8)  # Reduced max_size for better performance
    end_time = time.time()
    
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
    
    # Print all subgraphs in the summary
    classifier.print_subgraph_summary(show_all=True)

if __name__ == "__main__":
    main()