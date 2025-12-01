"""GraphSAGE-based recommendation model for RecSysFood.

This module implements a heterogeneous graph neural network for collaborative filtering
using user-recipe interactions.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Model save paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "trained"
MODEL_PATH = MODEL_DIR / "graphsage_model.pt"
DATA_PATH = MODEL_DIR / "graphsage_data.pt"
RECIPE_DB_PATH = MODEL_DIR / "graphsage_recipe_db.pt"
USER_MAPPING_PATH = MODEL_DIR / "graphsage_user_mapping.pt"
SCALERS_PATH = MODEL_DIR / "graphsage_scalers.pt"

# Define which features are continuous (need normalization) vs binary
CONTINUOUS_RECIPE_FEATURES = [
    "calories",
    "fatcontent",
    "proteincontent",
    "sugarcontent",
    "preptime_min",
    "totaltime_min",
]

BINARY_RECIPE_FEATURES = [
    "is_vegan",
    "is_vegetarian",
    "contains_pork",
    "contains_alcohol",
    "contains_gluten",
    "contains_nuts",
    "contains_dairy",
    "contains_egg",
    "contains_fish",
    "contains_soy",
    "is_breakfast_brunch",
    "is_dessert",
]


def pre_processing(
    user_preference_db: pd.DataFrame,
    recipe_db: pd.DataFrame,
    interaction_db: pd.DataFrame,
    recipe_scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
):
    """Process datasets to prepare for graph construction.

    Args:
        user_preference_db: DataFrame with user preferences (user_id, meal_types, calorie_goal, protein_goal)
        recipe_db: DataFrame with recipe features
        interaction_db: DataFrame with user-recipe interactions
        recipe_scaler: Optional pre-fitted scaler for recipe features (for inference)
        fit_scaler: Whether to fit a new scaler (True for training, False for inference)

    Returns:
        Tuple of (processed user_preference_db, processed recipe_db, processed interaction_db, recipe_scaler)
    """
    # Recipe processing - use lowercase column names matching database schema
    recipe_features = ["recipeid"] + CONTINUOUS_RECIPE_FEATURES + BINARY_RECIPE_FEATURES

    # Filter only existing columns
    available_features = [f for f in recipe_features if f in recipe_db.columns]
    recipe_db = recipe_db[available_features].copy()

    # Normalize continuous recipe features using StandardScaler
    available_continuous = [f for f in CONTINUOUS_RECIPE_FEATURES if f in recipe_db.columns]
    if available_continuous:
        if fit_scaler:
            recipe_scaler = StandardScaler()
            # Handle NaN values before scaling
            recipe_db[available_continuous] = recipe_db[available_continuous].fillna(0)
            recipe_db[available_continuous] = recipe_scaler.fit_transform(
                recipe_db[available_continuous]
            )
        elif recipe_scaler is not None:
            recipe_db[available_continuous] = recipe_db[available_continuous].fillna(0)
            recipe_db[available_continuous] = recipe_scaler.transform(
                recipe_db[available_continuous]
            )

    # Binary features are already 0/1, no scaling needed
    available_binary = [f for f in BINARY_RECIPE_FEATURES if f in recipe_db.columns]
    for col in available_binary:
        recipe_db[col] = recipe_db[col].fillna(0).astype(float)

    # User preference processing
    user_preference_db = user_preference_db.copy()

    # Encode categorical columns if they exist
    categorical_cols = ["meal_types", "calorie_goal", "protein_goal"]
    for col in categorical_cols:
        if col in user_preference_db.columns:
            le = LabelEncoder()
            user_preference_db[col] = le.fit_transform(user_preference_db[col].astype(str))

    # Interaction processing - drop interaction_id if present
    interaction_db = interaction_db.copy()
    if "interaction_id" in interaction_db.columns:
        interaction_db = interaction_db.drop("interaction_id", axis=1)

    return user_preference_db, recipe_db, interaction_db, recipe_scaler


def BuildGraph(user_preference_db: pd.DataFrame, recipe_db: pd.DataFrame, interaction_db: pd.DataFrame):
    """Build a bipartite graph from user preferences and recipes.

    Args:
        user_preference_db: Processed user preferences DataFrame
        recipe_db: Processed recipes DataFrame
        interaction_db: Processed interactions DataFrame

    Returns:
        NetworkX graph with user and recipe nodes
    """
    G = nx.Graph()

    # Define users preference node
    for _, row in user_preference_db.iterrows():
        node_id = str(row["user_id"])
        attrs = row.drop("user_id").to_dict()
        attrs["node_type"] = "user"
        G.add_node(f"u_{node_id}", bipartite=0, **attrs)

    # Define recipe node
    for _, row in recipe_db.iterrows():
        node_id = row["recipeid"]
        attrs = row.drop("recipeid").to_dict()
        attrs["node_type"] = "recipe"
        G.add_node(f"r_{node_id}", bipartite=1, **attrs)

    # Define edges from interactions
    for _, row in interaction_db.iterrows():
        user_node = f"u_{row['user_id']}"
        recipe_node = f"r_{row['recipe_id']}"
        # Only add edge if both nodes exist
        if user_node in G.nodes() and recipe_node in G.nodes():
            G.add_edge(user_node, recipe_node, rating=row["rating"])

    return G


def Graph_to_PyG(G, user_preference_db: pd.DataFrame, recipe_db: pd.DataFrame):
    """Convert NetworkX graph to PyTorch Geometric HeteroData.

    Args:
        G: NetworkX graph
        user_preference_db: Processed user preferences DataFrame
        recipe_db: Processed recipes DataFrame

    Returns:
        Tuple of (HeteroData, user_dict, recipe_dict) for mapping IDs
    """
    # Map node IDs
    unique_u = user_preference_db["user_id"].astype(str).unique()
    unique_r = recipe_db["recipeid"].unique()

    user_dict = {f"u_{str(u)}": i for i, u in enumerate(unique_u)}
    recipe_dict = {f"r_{str(r)}": i for i, r in enumerate(unique_r)}

    # Create features matrix
    users_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "user"]
    recipe_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "recipe"]

    num_recipe_features = [
        "calories",
        "fatcontent",
        "proteincontent",
        "sugarcontent",
        "preptime_min",
        "totaltime_min",
        "is_vegan",
        "is_vegetarian",
        "contains_pork",
        "contains_alcohol",
        "contains_gluten",
        "contains_nuts",
        "contains_dairy",
        "contains_egg",
        "contains_fish",
        "contains_soy",
        "is_breakfast_brunch",
        "is_dessert",
    ]

    # Create pyG data object
    data = HeteroData()

    # Convert Users nodes and features to pyG data object
    user_features = []
    for u in users_nodes:
        f = list(G.nodes[u].values())
        f = [float(v) if isinstance(v, (int, float, bool)) else 0.0 for v in f if v != "user"]
        user_features.append(f)

    if user_features:
        x_user = torch.tensor(user_features, dtype=torch.float)
        data["user"].x = x_user
    else:
        # Create dummy features if no users
        data["user"].x = torch.zeros((len(unique_u), 3), dtype=torch.float)

    # Convert recipes nodes and features to pyG data object
    recipe_feature = []
    for r in recipe_nodes:
        f = []
        for feat in num_recipe_features:
            val = G.nodes[r].get(feat, 0.0)
            f.append(float(val) if val is not None else 0.0)
        recipe_feature.append(f)

    if recipe_feature:
        x_recipe = torch.tensor(recipe_feature, dtype=torch.float)
        data["recipe"].x = x_recipe
    else:
        data["recipe"].x = torch.zeros((len(unique_r), len(num_recipe_features)), dtype=torch.float)

    # Convert edges to pyG data object
    edge_list = []
    for u, r, _ in G.edges(data=True):
        if str(u).startswith("u_") and str(r).startswith("r_"):
            if u in user_dict and r in recipe_dict:
                edge_list.append([user_dict[u], recipe_dict[r]])
        elif str(u).startswith("r_") and str(r).startswith("u_"):
            if r in user_dict and u in recipe_dict:
                edge_list.append([user_dict[r], recipe_dict[u]])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        data["user", "rating", "recipe"].edge_index = edge_index
        data = T.ToUndirected()(data)
    else:
        # Create empty edge index if no edges
        data["user", "rating", "recipe"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["recipe", "rev_rating", "user"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    return data, user_dict, recipe_dict


class GraphSAGELinkPredictor(torch.nn.Module):
    """GraphSAGE-based link prediction model for user-recipe recommendations."""

    def __init__(self, user_dim: int, recipe_dim: int, hidden_channels: int, out_channels: int):
        super().__init__()

        # Project both feature types -> hidden_dim
        self.user_lin = nn.Linear(user_dim, hidden_channels)
        self.recipe_lin = nn.Linear(recipe_dim, hidden_channels)

        # Batch normalization after initial projection
        self.user_bn_input = nn.BatchNorm1d(hidden_channels)
        self.recipe_bn_input = nn.BatchNorm1d(hidden_channels)

        # Define heterogeneous SAGEConv layers
        self.conv1 = HeteroConv(
            {
                ("user", "rating", "recipe"): SAGEConv(hidden_channels, hidden_channels),
                ("recipe", "rev_rating", "user"): SAGEConv(hidden_channels, hidden_channels),
            }
        )

        # Batch normalization after first conv layer
        self.user_bn1 = nn.BatchNorm1d(hidden_channels)
        self.recipe_bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = HeteroConv(
            {
                ("user", "rating", "recipe"): SAGEConv(hidden_channels, out_channels),
                ("recipe", "rev_rating", "user"): SAGEConv(hidden_channels, out_channels),
            }
        )

        # Batch normalization after second conv layer
        self.user_bn2 = nn.BatchNorm1d(out_channels)
        self.recipe_bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(p=0.5)

    def Encoder(self, x_dict, edge_index_dict):
        """Encode nodes using GraphSAGE convolutions with batch normalization."""
        # Initial projection with batch norm
        x_dict = {
            "user": self.user_bn_input(self.user_lin(x_dict["user"])),
            "recipe": self.recipe_bn_input(self.recipe_lin(x_dict["recipe"])),
        }
        x_dict = {k: nn.functional.relu(v) for k, v in x_dict.items()}

        # First conv layer with batch norm
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {
            "user": self.user_bn1(x_dict["user"]),
            "recipe": self.recipe_bn1(x_dict["recipe"]),
        }
        x_dict = {k: nn.functional.relu(v) for k, v in x_dict.items()}
        x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        # Second conv layer with batch norm
        z = self.conv2(x_dict, edge_index_dict)
        z = {
            "user": self.user_bn2(z["user"]),
            "recipe": self.recipe_bn2(z["recipe"]),
        }

        return z

    def Decoder(self, z, edge_label_index):
        """Decode edge scores using normalized dot product."""
        src, _, dst = ("user", "rating", "recipe")

        z_src = z[src][edge_label_index[0]]
        z_dest = z[dst][edge_label_index[1]]

        # L2 normalization of the embeddings for cosine similarity
        z_src = nn.functional.normalize(z_src, p=2, dim=1)
        z_dest = nn.functional.normalize(z_dest, p=2, dim=1)

        # Calculate the dot product for each edge (cosine similarity)
        scores = (z_src * z_dest).sum(dim=1)

        return scores


def training(model, train_data, optimizer, loss):
    """Train the model for one epoch."""
    model.train()

    edge_idx = train_data["user", "rating", "recipe"].edge_index
    pos_edge_lab = train_data["user", "rating", "recipe"].pos_edge_label_index

    # Check if we have edges to train on
    if pos_edge_lab.size(1) == 0:
        return 0.0

    nbr_user = train_data["user"].num_nodes
    nbr_recipe = train_data["recipe"].num_nodes

    neg_edge_lab = negative_sampling(
        edge_index=edge_idx,
        num_nodes=(nbr_user, nbr_recipe),
        num_neg_samples=pos_edge_lab.size(1),
    )

    edge_label_idx = torch.cat([pos_edge_lab, neg_edge_lab], dim=1).long()

    z = model.Encoder(
        x_dict={"user": train_data["user"].x, "recipe": train_data["recipe"].x},
        edge_index_dict={
            ("user", "rating", "recipe"): train_data["user", "rating", "recipe"].edge_index,
            ("recipe", "rev_rating", "user"): train_data["recipe", "rev_rating", "user"].edge_index,
        },
    )

    scores = model.Decoder(z, edge_label_idx)

    pos_labels = torch.ones(pos_edge_lab.size(1))
    neg_labels = torch.zeros(neg_edge_lab.size(1))
    edge_labels = torch.cat([pos_labels, neg_labels], dim=0)

    loss_fct = loss(scores, edge_labels)
    optimizer.zero_grad()
    loss_fct.backward()
    optimizer.step()

    return loss_fct.item()


@torch.no_grad()
def test(model, val_test_data):
    """Test the model and return AUC scores."""
    model.eval()

    edge_idx = val_test_data["user", "rating", "recipe"].edge_index
    pos_edge_lab = val_test_data["user", "rating", "recipe"].pos_edge_label_index

    if pos_edge_lab.size(1) == 0:
        return 0.5, 0.5  # Return baseline scores if no edges

    nbr_user = val_test_data["user"].num_nodes
    nbr_recipe = val_test_data["recipe"].num_nodes

    neg_edge_lab = negative_sampling(
        edge_index=edge_idx,
        num_nodes=(nbr_user, nbr_recipe),
        num_neg_samples=pos_edge_lab.size(1),
    )

    edge_label_index = torch.cat([pos_edge_lab, neg_edge_lab], dim=1).long()

    z = model.Encoder(
        x_dict={"user": val_test_data["user"].x, "recipe": val_test_data["recipe"].x},
        edge_index_dict={
            ("user", "rating", "recipe"): val_test_data["user", "rating", "recipe"].edge_index,
            ("recipe", "rev_rating", "user"): val_test_data["recipe", "rev_rating", "user"].edge_index,
        },
    )

    scores = model.Decoder(z, edge_label_index)

    pos_labels = torch.ones(pos_edge_lab.size(1))
    neg_labels = torch.zeros(neg_edge_lab.size(1))
    edge_labels = torch.cat([pos_labels, neg_labels], dim=0)

    if edge_labels.numel() == 0:
        return 0.5, 0.5

    auc = roc_auc_score(edge_labels.numpy(), scores.numpy())
    auc_pr = average_precision_score(edge_labels.numpy(), scores.numpy())

    return auc, auc_pr


def Train_GNN(user_preference_db: pd.DataFrame, recipe_db: pd.DataFrame, interaction_db: pd.DataFrame, epochs: int = 15):
    """Train the GraphSAGE model.

    Args:
        user_preference_db: DataFrame with user preferences
        recipe_db: DataFrame with recipe features
        interaction_db: DataFrame with user-recipe interactions
        epochs: Number of training epochs

    Returns:
        Tuple of (data, model, user_dict, recipe_dict, recipe_scaler)
    """
    # Process datasets with feature normalization (fit_scaler=True for training)
    user_preference_db, recipe_db_processed, interaction_db, recipe_scaler = pre_processing(
        user_preference_db, recipe_db, interaction_db, fit_scaler=True
    )

    # Build Bipartite Graph
    G = BuildGraph(user_preference_db, recipe_db_processed, interaction_db)

    # Convert the Graph into numerical data
    data, user_dict, recipe_dict = Graph_to_PyG(G, user_preference_db, recipe_db_processed)

    # Check if we have enough edges
    edge_count = data["user", "rating", "recipe"].edge_index.size(1)
    if edge_count < 10:
        print(f"Warning: Only {edge_count} edges in graph. Model may not train well.")
        # Return untrained model for inference
        user_dim = data["user"].x.size(1) if data["user"].x.size(0) > 0 else 3
        recipe_dim = data["recipe"].x.size(1) if data["recipe"].x.size(0) > 0 else 18
        model = GraphSAGELinkPredictor(
            user_dim=user_dim, recipe_dim=recipe_dim, hidden_channels=64, out_channels=32
        )
        return data, model, user_dict, recipe_dict, recipe_scaler

    # Split the data
    transform = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.05,
        is_undirected=True,
        add_negative_train_samples=False,
        split_labels=True,
        edge_types=[("user", "rating", "recipe")],
        rev_edge_types=[("recipe", "rev_rating", "user")],
    )
    train_data, val_data, test_data = transform(data)

    # Initialize model
    user_dim = train_data["user"].x.size(1)
    recipe_dim = train_data["recipe"].x.size(1)

    model = GraphSAGELinkPredictor(
        user_dim=user_dim, recipe_dim=recipe_dim, hidden_channels=64, out_channels=32
    )

    Adam_opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fct = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    for i in range(epochs):
        loss_value = training(model, train_data, optimizer=Adam_opt, loss=loss_fct)
        val_auc, val_ap = test(model, val_data)
        print(f"Epoch {i + 1}/{epochs} - Loss: {loss_value:.4f} - Val AUC: {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc

    # Test the model
    auc, auc_pr = test(model, test_data)
    print(f"Final test AUC score: {auc:.4f}")
    print(f"Final test AP score: {auc_pr:.4f}")

    return data, model, user_dict, recipe_dict, recipe_scaler


def Recommendation(model, data, user_id: str, recipe_db: pd.DataFrame, user_dict: dict, recipe_dict: dict, top_k: int = 15):
    """Generate top-k recommendations for a user.

    Args:
        model: Trained GraphSAGE model
        data: HeteroData graph
        user_id: User UUID string
        recipe_db: Original recipe DataFrame (not processed)
        user_dict: Mapping from user node IDs to indices
        recipe_dict: Mapping from recipe node IDs to indices
        top_k: Number of recommendations to return

    Returns:
        List of recommended recipe IDs with scores
    """
    model.eval()

    user_node = f"u_{user_id}"

    # Check if user exists in graph
    if user_node not in user_dict:
        print(f"User {user_id} not found in graph. Returning empty recommendations.")
        return []

    user_idx = user_dict[user_node]

    with torch.no_grad():
        out = model.Encoder(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)

    recipe_emb = out["recipe"]
    num_recipes = recipe_emb.size(0)

    # Create edge_label_index for all recipes
    edge_label_index = torch.stack(
        [
            torch.full((num_recipes,), user_idx, dtype=torch.long),
            torch.arange(num_recipes, dtype=torch.long),
        ]
    )

    scores = model.Decoder(out, edge_label_index)

    # Mask already rated recipes
    u, r = data["user", "rating", "recipe"].edge_index
    already_rated = r[u == user_idx].tolist()
    scores[already_rated] = -1e9

    # Get top-k recommendations
    top_scores, top_indices = scores.topk(min(top_k, num_recipes))
    recommended_indices = top_indices.tolist()
    recommended_scores = top_scores.tolist()

    # Convert graph indices to original recipe IDs
    idx_to_recipe = {v: k for k, v in recipe_dict.items()}

    recommendations = []
    for idx, score in zip(recommended_indices, recommended_scores):
        if idx in idx_to_recipe:
            recipe_node = idx_to_recipe[idx]
            recipe_id = int(recipe_node.split("_")[1])
            recommendations.append({"recipe_id": recipe_id, "score": float(score)})

    return recommendations


def save_model(
    model,
    data,
    recipe_db: pd.DataFrame,
    user_dict: dict,
    recipe_dict: dict,
    recipe_scaler: Optional[StandardScaler] = None,
):
    """Save trained model and data to disk.

    Args:
        model: Trained GraphSAGE model
        data: HeteroData graph
        recipe_db: Recipe DataFrame
        user_dict: User ID mapping
        recipe_dict: Recipe ID mapping
        recipe_scaler: Fitted StandardScaler for recipe features
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(data, DATA_PATH)
    torch.save(recipe_db.to_dict(), RECIPE_DB_PATH)
    torch.save({"user_dict": user_dict, "recipe_dict": recipe_dict}, USER_MAPPING_PATH)

    # Save scaler parameters if available
    if recipe_scaler is not None:
        scaler_params = {
            "mean_": recipe_scaler.mean_.tolist(),
            "scale_": recipe_scaler.scale_.tolist(),
            "var_": recipe_scaler.var_.tolist(),
            "n_features_in_": recipe_scaler.n_features_in_,
        }
        torch.save(scaler_params, SCALERS_PATH)

    print(f"Model saved to {MODEL_PATH}")


def load_model():
    """Load trained model and data from disk.

    Returns:
        Tuple of (model, data, recipe_db, user_dict, recipe_dict, recipe_scaler) or None if not found
    """
    if not all(p.exists() for p in [MODEL_PATH, DATA_PATH, RECIPE_DB_PATH, USER_MAPPING_PATH]):
        print("Model files not found. Please train the model first.")
        return None

    data = torch.load(DATA_PATH, weights_only=False)
    recipe_db_dict = torch.load(RECIPE_DB_PATH, weights_only=False)
    recipe_db = pd.DataFrame(recipe_db_dict)
    mappings = torch.load(USER_MAPPING_PATH, weights_only=False)

    user_dict = mappings["user_dict"]
    recipe_dict = mappings["recipe_dict"]

    # Load scaler if available
    recipe_scaler = None
    if SCALERS_PATH.exists():
        scaler_params = torch.load(SCALERS_PATH, weights_only=False)
        recipe_scaler = StandardScaler()
        recipe_scaler.mean_ = np.array(scaler_params["mean_"])
        recipe_scaler.scale_ = np.array(scaler_params["scale_"])
        recipe_scaler.var_ = np.array(scaler_params["var_"])
        recipe_scaler.n_features_in_ = scaler_params["n_features_in_"]

    # Reconstruct model
    user_dim = data["user"].x.size(1)
    recipe_dim = data["recipe"].x.size(1)

    model = GraphSAGELinkPredictor(
        user_dim=user_dim, recipe_dim=recipe_dim, hidden_channels=64, out_channels=32
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    print(f"Model loaded from {MODEL_PATH}")
    return model, data, recipe_db, user_dict, recipe_dict, recipe_scaler


def model_exists() -> bool:
    """Check if a trained model exists."""
    return all(p.exists() for p in [MODEL_PATH, DATA_PATH, RECIPE_DB_PATH, USER_MAPPING_PATH])


def get_recommendations_for_user(user_id: str, top_k: int = 15) -> list[dict]:
    """Get recommendations for a user using pre-trained model.

    This is the main inference function to be called from the API.

    Args:
        user_id: User UUID string
        top_k: Number of recommendations to return

    Returns:
        List of dicts with recipe_id and score, or empty list if model not available
    """
    loaded = load_model()
    if loaded is None:
        return []

    model, data, recipe_db, user_dict, recipe_dict, _recipe_scaler = loaded
    return Recommendation(model, data, user_id, recipe_db, user_dict, recipe_dict, top_k)
