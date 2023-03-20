import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, auc
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def import_graph(filepath):
    """
    Import a graph from a txt file
    :param filepath: filepath of the text file
    :return: G : graph from netwrokx library
    """
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


def generate_random_walk(graph, root, L):
    """
    :param graph: networkx graph
    :param root: the node where the random walk starts
    :param L: the length of the walk
    :return walk: list of the nodes visited by the random walk
    """
    walk = [root]
    while len(walk) < L:
        current_node = walk[-1]
        candidates = list(nx.neighbors(graph, current_node))
        next = np.random.choice(candidates)
        walk.append(next)

    return walk


def deep_walk(graph, N, L):
    """
    :param graph: networkx graph
    :param N: the number of walks for each node
    :param L: the walk length
    :return walks: the list of walks
    """
    walks = []
    np.random.seed(4)  # fix random seed to obtain same random shuffling when repeating experiment
    nodes = list(graph.nodes)

    for _ in tqdm(range(N)):
        np.random.shuffle(
            nodes)  # shuffle the ordering of nodes, it helps speed up the convergence of stochastic gradient descent
        for node in nodes:
            # generate a random walk from the current visited node
            walk = generate_random_walk(graph, node, L)
            walks.append(walk)
    return walks


def generate_samples(graph, train_set_ratio):
    """
    Graph pre-processing step required to perform supervised link prediction
    Create training and test sets
    """

    # --- Step 0: The graph must be connected ---
    if nx.is_connected(graph) is not True:
        raise ValueError("The graph contains more than one connected component!")

    # --- Step 1: Generate positive edge samples for testing set ---
    residual_g = graph.copy()
    test_pos_samples = []

    # Store the shuffled list of current edges of the graph
    edges = list(residual_g.edges())
    np.random.shuffle(edges)

    # Define number of positive test samples desired
    test_set_size = int((1.0 - train_set_ratio) * graph.number_of_edges())
    train_set_size = graph.number_of_edges() - test_set_size
    num_of_pos_test_samples = 0

    # Remove random edges from the graph, leaving it connected
    # Fill in the blanks
    for edge in edges:

        # Remove the edge
        residual_g.remove_edge(edge[0], edge[1])

        # Add the removed edge to the positive sample list if the network is still connected
        if nx.is_connected(residual_g):
            num_of_pos_test_samples += 1
            test_pos_samples.append(edge)
        # Otherwise, re-add the edge to the network
        else:
            residual_g.add_edge(edge[0], edge[1])

        # If we have collected enough number of edges for testing set, we can terminate the loop
        if num_of_pos_test_samples == test_set_size:
            break

    # Check if we have the desired number of positive samples for testing set
    if num_of_pos_test_samples != test_set_size:
        raise ValueError("Enough positive edge samples could not be found!")

    # --- Step 2: Generate positive edge samples for training set ---
    # The remaining edges are simply considered for positive samples of the training set
    train_pos_samples = list(residual_g.edges())

    # --- Step 3: Generate the negative samples for testing and training sets ---
    # Fill in the blanks
    non_edges = list(nx.non_edges(graph))
    np.random.shuffle(non_edges)

    train_neg_samples = non_edges[:train_set_size]
    test_neg_samples = non_edges[train_set_size:train_set_size + test_set_size]

    # --- Step 4: Combine sample lists and create corresponding labels ---
    # For training set
    train_samples = train_pos_samples + train_neg_samples
    train_labels = [1 for _ in train_pos_samples] + [0 for _ in train_neg_samples]
    # For testing set
    test_samples = test_pos_samples + test_neg_samples
    test_labels = [1 for _ in test_pos_samples] + [0 for _ in test_neg_samples]

    return residual_g, train_samples, train_labels, test_samples, test_labels


def edge_prediction(node2embedding, train_samples, test_samples, train_labels, test_labels, feature_func=None,
                    plot_roc=True):
    # --- Construct feature vectors for edges ---
    if feature_func is None:
        feature_func = lambda x, y: abs(x - y)

    # Fill in the blanks
    train_features = [feature_func(node2embedding[edge[0]], node2embedding[edge[1]]) for edge in train_samples]
    test_features = [feature_func(node2embedding[edge[0]], node2embedding[edge[1]]) for edge in test_samples]

    # --- Build the model and train it ---
    # Fill in the blanks
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)

    train_preds = clf.predict_proba(train_features)[:, 1]
    test_preds = clf.predict_proba(test_features)[:, 1]

    # --- Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from predictions ---
    # Fill in the blanks
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    roc_auc = auc(fpr, tpr)

    if not plot_roc:
        return roc_auc

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkred', label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc


