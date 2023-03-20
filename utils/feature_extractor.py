import numpy as np
from tqdm import tqdm
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def extract_node_features(df_node_info, node_id):
    """

    :param df_node_info: dataframe where node information is stored
    :param node_id: The id of the node
    :return:
    """
    return df_node_info[df_node_info.node_index == node_id]


def extract_graph_features(model, df):
    """
    Extract features embedding of a graph
    :param model: Model used for embedding
    :param train_df: dataframe containing source and target nodes
    :return: edge_features: array of all features
    """
    # Generate edge features
    edge_features = []
    for i, row in df.iterrows():
        source = row['source']
        target = row['target']
        features = np.concatenate([model.wv[str(source)], model.wv[str(target)]])
        edge_features.append(features)
    edge_features = np.array(edge_features)
    return edge_features


def get_relevant_node_features(node_info_df, n_components=128):
    """

    :param node_info_df: dataframe of the node features (node_information.csv)
    :param n_components: number of components of the reduced space
    :return: X_reduced_df: dataset projected on the reduced space
    """
    # Split node index and node features
    node_index = node_info_df.iloc[:, :1].values
    X = node_info_df.iloc[:, 1:].values

    # Scale the features using min-max scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit the PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Project node features onto reduced space
    X_reduced = pca.transform(X)

    # rebuild the dataframe
    X_reduced_df = pd.DataFrame(np.concatenate([node_index, X_reduced], axis=1))
    X_reduced_df = X_reduced_df.rename(columns={0: 'node_index'})

    return X_reduced_df


def get_combined_features(X_reduced_df, edge_features, df, is_train_dataset=True):
    # Initialize lists where we store source node info and target node info
    X_node_source = []
    X_node_target = []
    for k in tqdm(range(df.shape[0])):
        # Node indexes for source and target
        temp = df[df.index == k][["source", "target"]].values[0]
        if len(X_node_source) == 0:
            X_node_source = np.array(extract_node_features(X_reduced_df, temp[0]))[0].reshape(1, -1)[:, 1:]
            X_node_target = np.array(extract_node_features(X_reduced_df, temp[1]))[0].reshape(1, -1)[:, 1:]
        else:
            X_node_source = np.concatenate(
                [X_node_source, np.array(extract_node_features(X_reduced_df, temp[0]))[0].reshape(1, -1)[:, 1:]],
                axis=0)
            X_node_target = np.concatenate(
                [X_node_target, np.array(extract_node_features(X_reduced_df, temp[1]))[0].reshape(1, -1)[:, 1:]],
                axis=0)

    # Concatenate the source and target features
    X_edges = np.concatenate([X_node_source, X_node_target], axis=1)

    # Concatenate features extracted from graph structure with node features
    combined_features = np.concatenate([edge_features, X_edges], axis=1)

    # Add the node indexes
    source_target = df[["source", "target"]].values
    features = np.concatenate([source_target, combined_features], axis=1)

    labels_train = None
    if is_train_dataset:
        # Extract labels
        labels_train = df[["label"]].values


    # Convert to dataframe
    X_train = pd.DataFrame(features)
    X_train = X_train.rename(columns={0: 'node_source', 1: 'node_target'})

    return X_train, labels_train
