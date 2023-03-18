import networkx as nx


def import_graph(filepath):
    # Create an empty graph
    G = nx.Graph()

    # Open the txt file for reading
    with open(filepath, "r") as file:
        # Read each line of the file
        for line in file:
            # Split the line into its three components (source, target, label)
            source, target, label = line.strip().split()
            # Add the nodes and edges to the graph
            if label == "1":
                G.add_edge(source, target)
    return G