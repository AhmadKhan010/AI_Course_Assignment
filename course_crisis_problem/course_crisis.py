def read_swap_requests(filename):
    # Reads the swap request file and returns a list of valid requests
    # Each request: [roll_number, current_course, current_section, desired]
    # Skips requests where current_section == desired_section for same course
    requests = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue  # Skip malformed lines
                roll_number, course_section, curr_section, desired = parts
                course = course_section[:-2]  # Extract course (e.g., 'CN' from 'CN-A')
                # Skip if current and desired sections are the same for the same course
                if curr_section == desired and desired.isalpha() and course_section == f"{course}-{desired}":
                    continue
                requests.append([roll_number, course_section, curr_section, desired])
    except FileNotFoundError:
        print("Error: Input file not found.")
    return requests

def get_batch(roll_number):
    # Extracts batch from roll number ('21i-9772' -> 21)
    return int(roll_number.split('i-')[0])

def custom_shuffle(items, seed):
    # Custom shuffle function without using random library
    # Uses a seed (e.g., sum of roll number digits) for pseudo-randomness
    n = len(items)
    for i in range(n - 1, 0, -1):
        # Generate pseudo-random index using seed
        j = (seed + i) % (i + 1)
        items[i], items[j] = items[j], items[i]
    return items

def group_and_shuffle_requests(requests):
    # Returns a prioritized list of requests (batch 21, then 22, then 23, then 24)
    batch_groups = {21: [], 22: [], 23: [], 24: []}
    
    # Group requests by batch
    for req in requests:
        batch = get_batch(req[0])
        if batch in batch_groups:
            batch_groups[batch].append(req)
    
    for batch in batch_groups:
        # Use sum of roll number digits as seed for shuffling
        seed = sum(int(c) for req in batch_groups[batch] for c in req[0] if c.isdigit())
        batch_groups[batch] = custom_shuffle(batch_groups[batch], seed)
    
    # Combine batches in priority order
    return batch_groups[21] + batch_groups[22] + batch_groups[23] + batch_groups[24]

def build_swap_graph(requests):
    # Builds a directed graph where edges represent possible swaps
    # Edge from A to B means A wants B's current section/course
    graph = {i: [] for i in range(len(requests))}
    for i, req_i in enumerate(requests):
        roll_i, curr_course_i, curr_section_i, desired_i = req_i
        for j, req_j in enumerate(requests):
            if i == j:
                continue
            roll_j, curr_course_j, curr_section_j, desired_j = req_j
            # Check if req_i wants req_j's current position
            if desired_i == curr_section_j and curr_course_i.split('-')[0] == curr_course_j.split('-')[0]:
                # Same course, specific section swap
                graph[i].append(j)
            elif desired_i == curr_course_j:
                # req_i wants req_j's course-section
                graph[i].append(j)
            elif desired_i == 'A' and curr_course_j == curr_course_i:
                # req_i is flexible with section
                graph[i].append(j)
    return graph

def remove_used_nodes(graph, used_nodes):
    # Creates a new graph excluding used nodes
    new_graph = {}
    for node in graph:
        if node not in used_nodes:
            new_graph[node] = [n for n in graph[node] if n not in used_nodes]
    return new_graph

def dfs_find_cycle(graph, start, visited, parent, cycle):
    # DFS to find a cycle in the swap graph
    visited[start] = True
    cycle.append(start)
    
    for neighbor in graph[start]:
        if not visited[neighbor]:
            parent[neighbor] = start
            if dfs_find_cycle(graph, neighbor, visited, parent, cycle):
                return True
        elif neighbor in cycle and neighbor != parent.get(start, -1):
            # Found a cycle
            cycle_start = cycle.index(neighbor)
            cycle[:] = cycle[cycle_start:]
            return True
    cycle.pop()
    return False

def find_swap_cycles_dfs(graph):
    # Finds all swap cycles using DFS
    visited = {i: False for i in graph}
    cycles = []
    for start in graph:
        if not visited[start]:
            cycle = []
            parent = {}
            if dfs_find_cycle(graph, start, visited, parent, cycle):
                cycles.append(cycle)
                # Mark all nodes in cycle as visited
                for node in cycle:
                    visited[node] = True
    return cycles

def bfs_find_swaps(graph):
    # BFS to find direct swaps (2-cycles) and longer cycles
    swaps = []
    visited = {i: False for i in graph}
    
    for start in graph:
        if visited[start]:
            continue
        queue = [(start, [start])]
        visited[start] = True
        
        while queue:
            node, path = queue.pop(0)
            for neighbor in graph[node]:
                if neighbor in path and len(path) >= 2:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    if cycle not in swaps:
                        swaps.append(cycle)
                    continue
                if not visited[neighbor]:
                    visited[neighbor] = True
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
    return swaps

def greedy_find_swaps(graph, requests):
    # Greedy heuristic to find direct swaps only (2-cycles)
    swaps = []
    used = set()
    
    # Look for direct swaps (A -> B, B -> A)
    for i in range(len(requests)):
        if i in used:
            continue
        for j in graph[i]:
            if j in used:
                continue
            if i in graph[j]:  # Mutual swap
                swaps.append([i, j])
                used.add(i)
                used.add(j)
    
    return swaps, used

def generate_swap_instructions(swaps, requests, algorithm_name):
    # Generates human-readable swap instructions and tabular log entries for a specific algorithm
    instructions = []
    table_rows = []
    swap_no = 0
    
    for cycle_idx, cycle in enumerate(swaps, 1):
        swap_type = f"{algorithm_name} Direct" if len(cycle) == 2 else f"{algorithm_name} Cycle {cycle_idx}"
        if len(cycle) == 2:
            # Direct swap
            i, j = cycle
            req_i = requests[i]
            req_j = requests[j]
            instruction = (
                f"Swap student {req_i[0]} from {req_i[1]} to {req_j[1]} "
                f"with student {req_j[0]} from {req_j[1]} to {req_i[1]}"
            )
            instructions.append(instruction)
            # Add table rows for direct swap
            swap_no += 1
            table_rows.append([
                str(swap_no),
                req_i[0],
                req_i[1],
                req_j[1],
                swap_type
            ])
            table_rows.append([
                str(swap_no),
                req_j[0],
                req_j[1],
                req_i[1],
                swap_type
            ])
        else:
            # Cycle swap
            cycle_desc = []
            for k in range(len(cycle)):
                i = cycle[k]
                j = cycle[(k + 1) % len(cycle)]
                req_i = requests[i]
                req_j = requests[j]
                cycle_desc.append(
                    f"student {req_i[0]} from {req_i[1]} to {req_j[1]}"
                )
                # Add table row for each pairwise swap in the cycle
                swap_no += 1
                table_rows.append([
                    str(swap_no),
                    req_i[0],
                    req_i[1],
                    req_j[1],
                    swap_type
                ])
            instructions.append("Cycle swap: " + "; ".join(cycle_desc))
    
    return instructions, table_rows

def count_students(swaps):
    # Counts total students involved in swaps
    return sum(len(cycle) for cycle in swaps)

def count_pairwise_swaps(swaps):
    # Counts total pairwise swaps (2-cycle = 1 swap, n-cycle = n swaps for n >= 3)
    return sum(1 if len(cycle) == 2 else len(cycle) for cycle in swaps)

def print_tabular_schedule(table_rows):
    # Prints a tabular schedule of swaps
    headers = ["Swap No.", "Roll Number", "From Course-Section", "To Course-Section", "Swap Type"]
    # Calculate maximum width for each column
    col_widths = [len(header) for header in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    
    # Print header
    header_row = " | ".join(
        f"{header:<{col_widths[i]}}" for i, header in enumerate(headers)
    )
    print("\nSwap Schedule:")
    print("-" * (len(header_row) + 4))
    print(f"| {header_row} |")
    print("-" * (len(header_row) + 4))
    
    # Print rows
    for row in table_rows:
        formatted_row = " | ".join(
            f"{cell:<{col_widths[i]}}" for i, cell in enumerate(row)
        )
        print(f"| {formatted_row} |")
    print("-" * (len(header_row) + 4))

def main():
    filename = "data1.txt"
    
    # Step 1: Read and filter swap requests
    requests = read_swap_requests(filename)
    if not requests:
        print("No valid swap requests found.")
        return
    
    # Step 2: Group and shuffle requests by batch
    prioritized_requests = group_and_shuffle_requests(requests)
    
    # Step 3: Build swap graph
    graph = build_swap_graph(prioritized_requests)
    used_nodes = set()
    
    # Step 4: Apply all algorithms sequentially on the modified graph
    # Greedy (direct swaps only)
    greedy_swaps, greedy_used = greedy_find_swaps(graph, prioritized_requests)
    greedy_instructions, greedy_table_rows = generate_swap_instructions(greedy_swaps, prioritized_requests, "Greedy")
    used_nodes.update(greedy_used)
    
    # DFS on remaining graph
    dfs_graph = remove_used_nodes(graph, used_nodes)
    dfs_swaps = find_swap_cycles_dfs(dfs_graph)
    dfs_instructions, dfs_table_rows = generate_swap_instructions(dfs_swaps, prioritized_requests, "DFS")
    dfs_used = set(node for cycle in dfs_swaps for node in cycle)
    used_nodes.update(dfs_used)
    
    # BFS on remaining graph
    bfs_graph = remove_used_nodes(graph, used_nodes)
    bfs_swaps = bfs_find_swaps(bfs_graph)
    bfs_instructions, bfs_table_rows = generate_swap_instructions(bfs_swaps, prioritized_requests, "BFS")
    
    # Step 5: Combine results and print output
    all_table_rows = greedy_table_rows + dfs_table_rows + bfs_table_rows
    total_swaps = count_pairwise_swaps(greedy_swaps) + count_pairwise_swaps(dfs_swaps) + count_pairwise_swaps(bfs_swaps)
    total_students = count_students(greedy_swaps) + count_students(dfs_swaps) + count_students(bfs_swaps)
    
    print("\n=== Swap Results ===")
    
    # Greedy results
    print("\nGreedy Algorithm (Direct Swaps):")
    if greedy_instructions:
        for idx, inst in enumerate(greedy_instructions, 1):
            print(f"{idx}. {inst}")
        print(f"Students satisfied by Greedy: {count_students(greedy_swaps)}")
    else:
        print("No swaps found.")
    
    # DFS results
    print("\nDFS Algorithm (on remaining graph):")
    if dfs_instructions:
        for idx, inst in enumerate(dfs_instructions, 1):
            print(f"{idx}. {inst}")
        print(f"Students satisfied by DFS: {count_students(dfs_swaps)}")
    else:
        print("No swaps found.")
    
    # BFS results
    print("\nBFS Algorithm (on remaining graph):")
    if bfs_instructions:
        for idx, inst in enumerate(bfs_instructions, 1):
            print(f"{idx}. {inst}")
        print(f"Students satisfied by BFS: {count_students(bfs_swaps)}")
    else:
        print("No swaps found.")
    
    # Combined tabular schedule
    if all_table_rows:
        print_tabular_schedule(all_table_rows)
    
    # Summary statistics
    print(f"\nTotal pairwise swaps completed: {total_swaps}")
    print(f"Total students satisfied: {total_students}")
    
    if not all_table_rows:
        print("\nNo valid swaps found.")

if __name__ == "__main__":
    main()