from collections import deque


def adjacency(nodes, edges):
    result = {v: set() for v in nodes}
    for child, parent in edges:
        if child not in result or parent not in result:
            raise ValueError("Edge endpoint outside node set")
        if child == parent:
            raise ValueError("Self edge is not a WordNet direct relation")
        result[child].add(parent)
    return result


def reachable(adj, start):
    seen, queue = set(), deque(sorted(adj[start]))
    while queue:
        node = queue.popleft()
        if node not in seen:
            seen.add(node)
            queue.extend(sorted(adj[node] - seen))
    return seen


def closure(nodes, edges):
    adj = adjacency(nodes, edges)
    result = []
    for node in sorted(nodes):
        ancestors = reachable(adj, node)
        if node in ancestors:
            raise ValueError("Expected a DAG, found a directed cycle")
        result.extend((node, parent) for parent in sorted(ancestors))
    return result


def answer(graph, spec):
    adj = adjacency(graph["nodes"], graph["edges"])
    task = spec["type"]
    if task == "reachability":
        return spec["target"] in reachable(adj, spec["source"])
    if task == "common_parents":
        return sorted(adj[spec["left"]] & adj[spec["right"]])
    if task == "exact_k":
        frontier = {spec["source"]}
        for _ in range(spec["k"]):
            frontier = {w for v in frontier for w in adj[v]}
        return sorted(frontier)
    raise ValueError(f"Unknown task {task}")


def path_length(graph, source, target):
    adj = adjacency(graph["nodes"], graph["edges"])
    queue, seen = deque([(source, 0)]), {source}
    while queue:
        node, depth = queue.popleft()
        if node == target:
            return depth
        for other in sorted(adj[node] - seen):
            seen.add(other)
            queue.append((other, depth + 1))
    return None


def traversal(graph, order, direction="outgoing"):
    nodes = sorted(graph["nodes"])
    if order == "id":
        return nodes
    adj = adjacency(nodes, graph["edges"])
    if direction == "weak":
        # Ordering only: preserve directed facts in the renderer.
        for u, v in graph["edges"]:
            adj[v].add(u)
    elif direction != "outgoing":
        raise ValueError("Unknown serialization traversal direction")
    seen, output = set(), []
    for root in nodes:
        if root in seen:
            continue
        frontier = deque([root])
        while frontier:
            node = frontier.popleft() if order == "bfs" else frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            output.append(node)
            children = sorted(adj[node] - seen, reverse=order == "dfs")
            frontier.extend(children)
    return output
