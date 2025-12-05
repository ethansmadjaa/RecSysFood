"""Content-based recommendation model using GraphSAGE with Link Prediction.

This module implements a hybrid recommendation system that:
1. Creates recipe embeddings using GraphSAGE trained with link prediction loss
2. Creates user embeddings using SentenceTransformer on user preferences
3. Projects user embeddings into recipe space using a trained UserRecipeAdapter
4. Recommends recipes based on user profile similarity and liked recipe similarity
"""

import os
import re
from pathlib import Path
from typing import Optional

import nltk
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sentence_transformers import SentenceTransformer
from torch.utils.data import TensorDataset, DataLoader
from utils.logger import logger

# Device selection: prefer MPS (Apple Silicon), then CUDA, then CPU
def get_device() -> torch.device:
    """Get the best available device for training."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# Download NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# Model save paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "trained"
MODEL_PATH = MODEL_DIR / "content_model.pt"
ADAPTER_PATH = MODEL_DIR / "user_adapter.pt"
RECIPE_EMBEDDINGS_PATH = MODEL_DIR / "recipe_embeddings.pt"
RECIPE_INTERACTION_PATH = MODEL_DIR / "recipe_interaction.pt"
RECIPE_MAPPING_PATH = MODEL_DIR / "recipe_mapping.pt"
USER_CACHE_PATH = MODEL_DIR / "user_cache.pt"

# Recipe features (lowercase to match database schema)
RECIPE_FEATURES = [
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

# SBERT model dimension (all-MiniLM-L6-v2)
SBERT_DIM = 384
GNN_OUT_DIM = 64


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class GraphSAGELinkPredictor(nn.Module):
    """GraphSAGE model trained with link prediction objective."""

    def __init__(self, input_dim: int, hidden_channels: int = 128, out_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode recipe features into embeddings."""
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass returns embeddings."""
        return self.encode(x, edge_index)

    def predict_link_score(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Predict link scores using dot product of embeddings."""
        src, dst = edge_index[0], edge_index[1]
        return (z[src] * z[dst]).sum(dim=-1)


class UserRecipeAdapter(nn.Module):
    """MLP adapter to project SBERT user embeddings into GNN recipe space."""

    def __init__(self, input_dim: int = SBERT_DIM, output_dim: int = GNN_OUT_DIM):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
def _create_recipe_interaction_graph(recipe_df: pd.DataFrame, n_neighbors: int = 10) -> pd.DataFrame:
    """Build recipe-recipe interaction graph based on feature similarity.

    Uses parallelized KNN with all available CPU cores for faster computation.

    Args:
        recipe_df: DataFrame with recipe features
        n_neighbors: Number of nearest neighbors per recipe

    Returns:
        DataFrame with columns [recipe_idx_1, recipe_idx_2] representing edges
    """
    print(f"Building recipe interaction graph for {len(recipe_df)} recipes...")
    n_recipes = len(recipe_df)

    available_features = [f for f in RECIPE_FEATURES if f in recipe_df.columns]
    print(f"Using {len(available_features)} features for similarity computation")
    feature_matrix = recipe_df[available_features].fillna(0).values.astype(np.float32)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Use all CPU cores for KNN computation
    n_jobs = -1  # -1 = use all available cores
    print(f"Initializing KNN with n_neighbors={min(n_neighbors, n_recipes)}, using all CPU cores...")
    knn = NearestNeighbors(
        n_neighbors=min(n_neighbors, n_recipes),
        metric="cosine",
        algorithm="brute",  
        n_jobs=n_jobs
    )
    print("Fitting KNN model...")
    knn.fit(feature_matrix)
    print("Computing nearest neighbors...")
    _, indices = knn.kneighbors(feature_matrix)
    print(f"KNN computation complete. Indices shape: {indices.shape}")

    # Vectorized edge creation instead of nested loops
    print("Creating edges from neighbor indices...")
    n_neighbors_actual = indices.shape[1]
    recipe_indices = np.repeat(np.arange(n_recipes), n_neighbors_actual - 1)
    neighbor_indices = indices[:, 1:].flatten()  # Skip first column (self)

    interactions = np.column_stack([recipe_indices, neighbor_indices])
    print(f"Created {len(interactions)} edges in interaction graph")

    return pd.DataFrame(interactions, columns=["recipe_idx_1", "recipe_idx_2"])


def _build_networkx_graph(recipe_df: pd.DataFrame, interaction_df: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from recipe data and interactions."""
    G = nx.Graph()
    available_features = [f for f in RECIPE_FEATURES if f in recipe_df.columns]

    for idx, row in recipe_df.iterrows():
        attrs = {feat: float(row.get(feat, 0.0)) for feat in available_features}
        attrs["node_type"] = "recipe"
        G.add_node(idx, **attrs)

    for _, row in interaction_df.iterrows():
        G.add_edge(row["recipe_idx_1"], row["recipe_idx_2"])

    return G


def _graph_to_pyg(G: nx.Graph) -> Data:
    """Convert NetworkX graph to PyTorch Geometric Data object."""
    recipe_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "recipe"]
    node_id_to_idx = {node_id: i for i, node_id in enumerate(recipe_nodes)}

    available_features = [f for f in RECIPE_FEATURES if f in G.nodes[recipe_nodes[0]]]

    features = []
    for node_id in recipe_nodes:
        attrs = G.nodes[node_id]
        f = [float(attrs.get(feat, 0.0)) for feat in available_features]
        features.append(f)

    x = torch.tensor(features, dtype=torch.float)

    edge_list = []
    for u, v in G.edges():
        if u in node_id_to_idx and v in node_id_to_idx:
            edge_list.append([node_id_to_idx[u], node_id_to_idx[v]])
            edge_list.append([node_id_to_idx[v], node_id_to_idx[u]])  # Undirected

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# =============================================================================
# GNN TRAINING WITH LINK PREDICTION
# =============================================================================

def _train_gnn_link_prediction(
    data: Data,
    model: GraphSAGELinkPredictor,
    epochs: int = 50,
    lr: float = 0.0005,
    verbose: bool = False
) -> GraphSAGELinkPredictor:
    """Train GraphSAGE model with link prediction loss (BCEWithLogitsLoss)."""
    # Move model and data to device
    model = model.to(DEVICE)
    data = data.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Generate embeddings
        z = model.forward(data.x, data.edge_index)

        # Positive edges (existing links)
        pos_edge_index = data.edge_index
        pos_scores = model.predict_link_score(z, pos_edge_index)

        # Negative edges (non-existing links)
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        neg_scores = model.predict_link_score(z, neg_edge_index)

        # Compute loss
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(pos_scores.size(0), device=DEVICE),
            torch.zeros(neg_scores.size(0), device=DEVICE)
        ])

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    return model


def _evaluate_gnn(
    data: Data,
    model: GraphSAGELinkPredictor,
    verbose: bool = False
) -> tuple[torch.Tensor, float, float]:
    """Evaluate GNN model and return embeddings with metrics."""
    # Ensure model and data are on the same device
    model = model.to(DEVICE)
    data = data.to(DEVICE)

    model.eval()
    with torch.no_grad():
        x = data.x
        edge_idx = data.edge_index

        # Generate embeddings
        z = model.forward(x, edge_idx)

        # Create test edges
        pos_edge_lab = data.pos_edge_label_index if hasattr(data, 'pos_edge_label_index') else edge_idx
        neg_edge_lab = negative_sampling(
            edge_index=edge_idx,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_lab.size(1)
        )

        edge_label_index = torch.cat([pos_edge_lab, neg_edge_lab], dim=1)
        scores = model.predict_link_score(z, edge_label_index)

        # Create labels
        pos_labels = torch.ones(pos_edge_lab.size(1), device=DEVICE)
        neg_labels = torch.zeros(neg_edge_lab.size(1), device=DEVICE)
        edge_labels = torch.cat([pos_labels, neg_labels], dim=0)

        # Compute metrics (move to CPU for sklearn)
        auc = roc_auc_score(edge_labels.cpu().numpy(), scores.cpu().numpy())
        auc_pr = average_precision_score(edge_labels.cpu().numpy(), scores.cpu().numpy())

        if verbose:
            print(f"AUC = {auc:.4f} -- AUC-PR = {auc_pr:.4f}")

        return z, auc, auc_pr


# =============================================================================
# USER ADAPTER TRAINING (Self-supervised)
# =============================================================================

def _create_recipe_profile_text(recipe_row: pd.Series) -> str:
    """Create synthetic user profile text from recipe attributes for adapter training."""
    # Meal type
    meal_type = []
    if recipe_row.get('is_breakfast_brunch'):
        meal_type.append("breakfast brunch")
    if recipe_row.get('is_dessert'):
        meal_type.append("dessert")
    meal_text = f"Meal type preference: {', '.join(meal_type) if meal_type else 'main course'}. "

    # Calorie goal
    calorie_cat = recipe_row.get('calorie_category', 'medium')
    calorie_text = f"Calorie goal: {calorie_cat} calories. "

    # Protein goal
    protein_cat = recipe_row.get('protein_category', 'medium')
    protein_text = f"Protein goal: {protein_cat} protein. "

    # Dietary restrictions
    restrictions = []
    if recipe_row.get('is_vegan'):
        restrictions.append("vegan")
    if recipe_row.get('is_vegetarian'):
        restrictions.append("vegetarian")
    if not recipe_row.get('contains_pork', True):
        restrictions.append("no pork")
    if not recipe_row.get('contains_alcohol', True):
        restrictions.append("no alcohol")
    if not recipe_row.get('contains_gluten', True):
        restrictions.append("gluten-free")
    if not recipe_row.get('contains_nuts', True):
        restrictions.append("nut-free")
    if not recipe_row.get('contains_dairy', True):
        restrictions.append("dairy-free")
    if not recipe_row.get('contains_egg', True):
        restrictions.append("egg-free")
    if not recipe_row.get('contains_fish', True):
        restrictions.append("no fish")
    if not recipe_row.get('contains_soy', True):
        restrictions.append("soy-free")

    restrictions_text = f"Dietary restrictions: {', '.join(restrictions) if restrictions else 'none'}. "

    # Total time
    total_time = recipe_row.get('totaltime_min', 60)
    time_text = f"Max total preparation time: {total_time} minutes."

    return meal_text + calorie_text + protein_text + restrictions_text + time_text


def _prepare_adapter_training_data(
    recipe_df: pd.DataFrame,
    recipe_gnn_embeddings: torch.Tensor,
    sample_size: int = 100000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare self-supervised training data for the UserRecipeAdapter."""
    if len(recipe_df) > sample_size:
        sample_df = recipe_df.sample(n=sample_size, random_state=42)
        sample_idx = sample_df.index.tolist()
        gnn_targets = recipe_gnn_embeddings[sample_idx]
    else:
        sample_df = recipe_df.reset_index(drop=True)
        gnn_targets = recipe_gnn_embeddings

    # Create synthetic profile texts
    descriptions = sample_df.apply(_create_recipe_profile_text, axis=1).tolist()

    # Encode with SBERT
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    description_embeddings = sbert_model.encode(
        descriptions,
        convert_to_tensor=True,
        batch_size=2048,
        show_progress_bar=True
    )

    return description_embeddings, gnn_targets


def _train_adapter(
    adapter: UserRecipeAdapter,
    data_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.0005,
    verbose: bool = False
) -> UserRecipeAdapter:
    """Train the UserRecipeAdapter with cosine embedding loss."""
    # Move adapter to device
    adapter = adapter.to(DEVICE)
    adapter.train()
    optimizer = optim.Adam(adapter.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch_user_embs, batch_target_embs in data_loader:
            # Move batch to device
            batch_user_embs = batch_user_embs.to(DEVICE)
            batch_target_embs = batch_target_embs.to(DEVICE)

            optimizer.zero_grad()

            # Positive pairs (target label = 1)
            target_labels = torch.ones(batch_user_embs.shape[0], device=DEVICE)

            # Project user embeddings
            predicted_recipe_vector = adapter(batch_user_embs)

            # Compute loss
            loss = criterion(predicted_recipe_vector, batch_target_embs, target_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose and epoch % 10 == 0:
            print(f"Adapter Epoch {epoch}: Average Loss = {total_loss / len(data_loader):.4f}")

    return adapter


# =============================================================================
# TEXT PROCESSING & USER EMBEDDING
# =============================================================================

def _clean_text(text: str) -> str:
    """Clean and normalize text for embedding."""
    stop_words = set(stopwords.words("english"))

    text = str(text).lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"^ ", "", text)
    text = re.sub(r" $", "", text)
    text = re.sub(r"_", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text


def _create_user_embedding(user_prefs: dict, sentence_model: SentenceTransformer) -> torch.Tensor:
    """Create user embedding from preferences using SentenceTransformer."""
    parts = []

    if "meal_types" in user_prefs and user_prefs["meal_types"]:
        meal_types = user_prefs["meal_types"]
        if isinstance(meal_types, list):
            parts.append(f"meal type {' '.join(meal_types)}")
        else:
            parts.append(f"meal type {meal_types}")

    if "calorie_goal" in user_prefs:
        parts.append(f"calorie goal {user_prefs['calorie_goal']}")

    if "protein_goal" in user_prefs:
        parts.append(f"protein goal {user_prefs['protein_goal']}")

    user_text = _clean_text(" ".join(parts))

    if not user_text.strip():
        user_text = "general food preference"

    user_embed = sentence_model.encode(user_text, convert_to_tensor=True)
    return user_embed


# =============================================================================
# RECOMMENDATION LOGIC
# =============================================================================

def _calculate_similarity(
    query_embed: torch.Tensor,
    target_embeds: torch.Tensor,
    adapter: Optional[UserRecipeAdapter] = None,
    k: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate cosine similarity and return top-k matches.

    If adapter is provided and dimensions don't match, uses adapter for projection.
    """
    query_embed = query_embed.clone().detach().float().cpu()
    target_embeds = target_embeds.cpu()

    # Project using adapter if dimensions don't match
    if query_embed.dim() == 1:
        target_dim = target_embeds.size(1)
        if query_embed.size(0) != target_dim:
            if adapter is not None:
                adapter.eval()
                with torch.no_grad():
                    query_embed = adapter(query_embed.unsqueeze(0)).squeeze(0)
            else:
                # Fallback: random projection (not recommended)
                projection = nn.Linear(query_embed.size(0), target_dim)
                with torch.no_grad():
                    query_embed = projection(query_embed)

    # Normalize for cosine similarity
    query_norm = query_embed / (query_embed.norm() + 1e-8)
    target_norm = target_embeds / (target_embeds.norm(dim=1, keepdim=True) + 1e-8)

    # Calculate scores
    scores = target_norm @ query_norm
    k = min(k, len(scores))
    topk = scores.topk(k)

    return topk.indices, topk.values


def _find_most_similar_liked_recipe(
    recipe_idx: int,
    liked_indices: list[int],
    recipe_embeddings: torch.Tensor,
    idx_to_recipe_id: dict,
    recipe_db: pd.DataFrame
) -> tuple[Optional[str], float]:
    """Find which liked recipe is most similar to a recommended recipe."""
    if not liked_indices:
        return None, 0.0

    rec_emb = recipe_embeddings[recipe_idx].unsqueeze(0)
    liked_embeds = recipe_embeddings[liked_indices]

    rec_norm = rec_emb / (rec_emb.norm() + 1e-8)
    liked_norm = liked_embeds / (liked_embeds.norm(dim=1, keepdim=True) + 1e-8)
    similarities = (liked_norm @ rec_norm.T).squeeze()

    if similarities.dim() == 0:
        best_idx = 0
        best_score = float(similarities)
    else:
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

    best_liked_global_idx = liked_indices[best_idx]
    best_liked_id = idx_to_recipe_id[best_liked_global_idx]

    liked_row = recipe_db[recipe_db["recipeid"] == best_liked_id]
    if not liked_row.empty:
        return liked_row.iloc[0].get("name", None), best_score

    return None, best_score


def _generate_reason(
    recipe_id: int,
    recipe_db: pd.DataFrame,
    user_prefs: dict,
    liked_recipe_name: Optional[str] = None,
    similarity_score: float = 0.0,
    is_liked_based: bool = False
) -> str:
    """Generate a human-readable reason for recommending a recipe."""
    reasons = []

    recipe_row = recipe_db[recipe_db["recipeid"] == recipe_id]
    if recipe_row.empty:
        return "Recommended for you"

    recipe = recipe_row.iloc[0]

    # 1. Similarity to liked recipe
    if is_liked_based and liked_recipe_name and similarity_score > 0.3:
        reasons.append(f'Similar to "{liked_recipe_name}"')

    # 2. Recipe-specific attributes
    total_time = recipe.get("totaltime_min")
    if total_time and total_time <= 15:
        reasons.append("Ready in under 15 min")
    elif total_time and total_time <= 30:
        reasons.append("Quick to prepare")

    if recipe.get("is_vegan"):
        reasons.append("Vegan")
    elif recipe.get("is_vegetarian"):
        reasons.append("Vegetarian")

    calories = recipe.get("calories")
    if calories and calories < 200:
        reasons.append("Low calorie")
    elif calories and calories > 600:
        reasons.append("Hearty meal")

    protein = recipe.get("proteincontent")
    if protein and protein > 30:
        reasons.append("High protein")

    if recipe.get("is_breakfast_brunch"):
        reasons.append("Breakfast option")
    if recipe.get("is_dessert"):
        reasons.append("Sweet treat")

    # 3. Match with user preferences
    meal_types = user_prefs.get("meal_types", [])
    if meal_types and not reasons:
        if recipe.get("is_breakfast_brunch") and "breakfast" in [m.lower() for m in meal_types]:
            reasons.append("Matches your breakfast preference")
        if recipe.get("is_dessert") and "dessert" in [m.lower() for m in meal_types]:
            reasons.append("Matches your dessert preference")

    # 4. Calorie/protein goals as fallback
    if not reasons:
        calorie_goal = user_prefs.get("calorie_goal", "").lower()
        calorie_cat = str(recipe.get("calorie_category", "")).lower()
        if calorie_goal and calorie_cat:
            if calorie_goal == "low" and calorie_cat == "low":
                reasons.append("Fits your calorie goal")
            elif calorie_goal == "high" and calorie_cat == "high":
                reasons.append("Fits your calorie goal")

        protein_goal = user_prefs.get("protein_goal", "").lower()
        protein_cat = str(recipe.get("protein_category", "")).lower()
        if protein_goal and protein_cat:
            if protein_goal == "high" and protein_cat == "high":
                reasons.append("High in protein")

    if reasons:
        return " · ".join(reasons[:2])
    return "Recommended for you"


# =============================================================================
# PUBLIC API
# =============================================================================

def Train_GNN(
    user_preference_db: pd.DataFrame,
    recipe_db: pd.DataFrame,
    interaction_db: pd.DataFrame,
    epochs: int = 50,
    adapter_epochs: int = 50,
    verbose: bool = True
):
    """Train the content-based recommendation model with link prediction.

    Args:
        user_preference_db: DataFrame with user preferences (for API compatibility)
        recipe_db: DataFrame with recipe features
        interaction_db: DataFrame with user-recipe interactions
        epochs: Number of GNN training epochs
        adapter_epochs: Number of adapter training epochs
        verbose: Whether to print training progress

    Returns:
        Tuple of (recipe_embeddings, model, recipe_id_to_idx, idx_to_recipe_id, data, adapter)
    """
    print(f"Training content-based model on {len(recipe_db)} recipes...")

    # Normalize column names
    recipe_db = recipe_db.copy()
    recipe_db.columns = recipe_db.columns.str.lower()
    print("recipe_db normalized")

    # Create recipe interaction graph
    interaction_graph = _create_recipe_interaction_graph(recipe_db)
    print("interaction_graph created")

    # Build NetworkX graph
    G = _build_networkx_graph(recipe_db.reset_index(drop=True), interaction_graph)
    print("G created")

    # Convert to PyG
    data = _graph_to_pyg(G)
    print("data created")

    if data.x.size(0) == 0:
        print("Warning: No recipe data to train on")
        return None, None, {}, {}, None, None

    # Normalize features
    scaler = StandardScaler()
    data_np = data.x.cpu().numpy()
    normalized_data = scaler.fit_transform(data_np)
    data.x = torch.tensor(normalized_data, dtype=torch.float)

    # Initialize and train GNN model
    input_dim = data.x.size(1)
    model = GraphSAGELinkPredictor(
        input_dim=input_dim,
        hidden_channels=128,
        out_channels=GNN_OUT_DIM
    )
    print(f"model initialized (device: {DEVICE})")

    if data.edge_index.size(1) > 0:
        model = _train_gnn_link_prediction(data, model, epochs=epochs, verbose=verbose)
        print("model trained with link prediction")

    # Generate embeddings (ensure data and model are on same device)
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    model.eval()
    with torch.no_grad():
        recipe_embeddings = model.encode(data.x, data.edge_index)
    # Move embeddings to CPU for storage and inference compatibility
    recipe_embeddings = recipe_embeddings.cpu()
    print(f"recipe_embeddings generated: {recipe_embeddings.size()}")

    # Train UserRecipeAdapter
    print("Training UserRecipeAdapter...")
    adapter = UserRecipeAdapter(input_dim=SBERT_DIM, output_dim=GNN_OUT_DIM)

    user_emb_sim, gnn_targets = _prepare_adapter_training_data(recipe_db, recipe_embeddings)
    adapter_dataset = TensorDataset(user_emb_sim, gnn_targets)
    adapter_loader = DataLoader(adapter_dataset, batch_size=2048, shuffle=True)

    adapter = _train_adapter(adapter, adapter_loader, epochs=adapter_epochs, verbose=verbose)
    # Move adapter back to CPU for saving
    adapter = adapter.cpu()
    print("adapter trained")

    # Move model and data back to CPU for saving/returning
    model = model.cpu()
    data = data.cpu()

    # Create mappings
    recipe_ids = recipe_db["recipeid"].tolist()
    recipe_id_to_idx = {rid: idx for idx, rid in enumerate(recipe_ids)}
    idx_to_recipe_id = {idx: rid for rid, idx in recipe_id_to_idx.items()}
    print("mappings created")

    print(f"Generated {recipe_embeddings.size(0)} recipe embeddings of dimension {recipe_embeddings.size(1)}")
    return recipe_embeddings, model, recipe_id_to_idx, idx_to_recipe_id, data, adapter


def save_model(
    model: GraphSAGELinkPredictor,
    data: Data,
    recipe_db: pd.DataFrame,
    recipe_id_to_idx: dict,
    idx_to_recipe_id: dict,
    adapter: Optional[UserRecipeAdapter] = None
):
    """Save trained model, adapter, and embeddings to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure model is on CPU before saving
    model = model.cpu()
    data = data.cpu()

    # Save GNN model state
    torch.save(model.state_dict(), MODEL_PATH)

    # Save adapter if provided
    if adapter is not None:
        adapter = adapter.cpu()
        torch.save(adapter.state_dict(), ADAPTER_PATH)

    # Generate and save embeddings
    model.eval()
    with torch.no_grad():
        recipe_embeddings = model.encode(data.x, data.edge_index)
    torch.save(recipe_embeddings.cpu(), RECIPE_EMBEDDINGS_PATH)

    # Save data for inference
    torch.save(data, RECIPE_INTERACTION_PATH)

    # Save mappings
    torch.save({
        "recipe_id_to_idx": recipe_id_to_idx,
        "idx_to_recipe_id": idx_to_recipe_id,
        "recipe_db": recipe_db.to_dict(),
    }, RECIPE_MAPPING_PATH)

    print(f"Model saved to {MODEL_PATH}")


def load_model():
    """Load trained model and embeddings from disk.

    Returns:
        Tuple of (model, recipe_embeddings, recipe_id_to_idx, idx_to_recipe_id, data, recipe_db, adapter) or None
    """
    if not model_exists():
        print("Model files not found. Please train the model first.")
        return None

    # Load data
    data = torch.load(RECIPE_INTERACTION_PATH, weights_only=False)
    mappings = torch.load(RECIPE_MAPPING_PATH, weights_only=False)
    recipe_embeddings = torch.load(RECIPE_EMBEDDINGS_PATH, weights_only=False)

    recipe_id_to_idx = mappings["recipe_id_to_idx"]
    idx_to_recipe_id = mappings["idx_to_recipe_id"]
    recipe_db = pd.DataFrame(mappings.get("recipe_db", {}))

    # Reconstruct GNN model
    input_dim = data.x.size(1)
    model = GraphSAGELinkPredictor(input_dim=input_dim, hidden_channels=128, out_channels=GNN_OUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # Load adapter if exists
    adapter = None
    if ADAPTER_PATH.exists():
        adapter = UserRecipeAdapter(input_dim=SBERT_DIM, output_dim=GNN_OUT_DIM)
        adapter.load_state_dict(torch.load(ADAPTER_PATH, weights_only=True))
        adapter.eval()

    print(f"Model loaded from {MODEL_PATH}")
    return model, recipe_embeddings, recipe_id_to_idx, idx_to_recipe_id, data, recipe_db, adapter


def model_exists() -> bool:
    """Check if a trained model exists."""
    return all(p.exists() for p in [MODEL_PATH, RECIPE_EMBEDDINGS_PATH, RECIPE_INTERACTION_PATH, RECIPE_MAPPING_PATH])


def user_exists_in_model(user_id: str) -> bool:
    """Check if we can generate recommendations for a user."""
    return model_exists()


def get_recommendations_for_user(
    user_id: str,
    top_k: int = 15,
    user_prefs: Optional[dict] = None,
    liked_recipe_ids: Optional[list[int]] = None,
    k_user: int = 500,
    k_recipe: int = 100
) -> list[dict]:
    """Get recommendations for a user using the content-based model with reranking.

    Args:
        user_id: User UUID string
        top_k: Number of recommendations to return
        user_prefs: Optional dict with user preferences (meal_types, calorie_goal, protein_goal)
        liked_recipe_ids: Optional list of recipe IDs the user has liked
        k_user: Number of user-based candidates to consider
        k_recipe: Number of recipe-based candidates for reranking

    Returns:
        List of dicts with recipe_id, score, and reason
    """
    loaded = load_model()
    if loaded is None:
        return []

    model, recipe_embeddings, recipe_id_to_idx, idx_to_recipe_id, data, recipe_db, adapter = loaded

    # Load sentence transformer for user embedding
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    if user_prefs is None:
        user_prefs = {}

    user_embed = _create_user_embedding(user_prefs, sentence_model)

    # Step 1: Get user-recipe recommendations
    k_user = min(k_user, len(recipe_embeddings))
    user_top_idx, user_top_scores = _calculate_similarity(
        user_embed, recipe_embeddings, adapter=adapter, k=k_user
    )
    user_top_idx = user_top_idx.tolist()
    user_top_scores = [float(s) for s in user_top_scores]

    # Step 2: If user has liked recipes, rerank based on recipe-recipe similarity
    if liked_recipe_ids and len(liked_recipe_ids) > 0:
        liked_indices = []
        for rid in liked_recipe_ids:
            if rid in recipe_id_to_idx:
                liked_indices.append(recipe_id_to_idx[rid])

        if liked_indices:
            # Find the best liked recipe for reranking
            best_liked_idx = None
            best_mean_score = -float("inf")

            for liked_idx in liked_indices:
                liked_emb = recipe_embeddings[liked_idx]
                candidate_embeds = recipe_embeddings[user_top_idx]
                _, scores = _calculate_similarity(liked_emb, candidate_embeds, k=k_recipe)

                mean_score = float(scores.mean())
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    best_liked_idx = liked_idx

            if best_liked_idx is not None:
                # Rerank using best liked recipe
                liked_emb = recipe_embeddings[best_liked_idx]
                candidate_embeds = recipe_embeddings[user_top_idx]
                top_local_idx, top_recipe_scores = _calculate_similarity(
                    liked_emb, candidate_embeds, k=k_recipe
                )

                recommendations = []
                for local_idx, recipe_score in zip(top_local_idx.tolist(), top_recipe_scores.tolist()):
                    global_idx = user_top_idx[local_idx]
                    recipe_id = idx_to_recipe_id[global_idx]

                    if recipe_id not in liked_recipe_ids:
                        liked_recipe_name, similarity_score = _find_most_similar_liked_recipe(
                            recipe_idx=global_idx,
                            liked_indices=liked_indices,
                            recipe_embeddings=recipe_embeddings,
                            idx_to_recipe_id=idx_to_recipe_id,
                            recipe_db=recipe_db
                        )

                        reason = _generate_reason(
                            recipe_id=recipe_id,
                            recipe_db=recipe_db,
                            user_prefs=user_prefs,
                            liked_recipe_name=liked_recipe_name,
                            similarity_score=similarity_score,
                            is_liked_based=True
                        )

                        # Combine user and recipe similarity scores
                        user_score = user_top_scores[local_idx] if local_idx < len(user_top_scores) else 0.0
                        combined_score = 0.5 * user_score + 0.5 * recipe_score

                        recommendations.append({
                            "recipe_id": recipe_id,
                            "score": float(combined_score),
                            "reason": reason
                        })

                        if len(recommendations) >= top_k:
                            break

                # Sort by combined score
                recommendations.sort(key=lambda x: x["score"], reverse=True)
                return recommendations[:top_k]

    # Fallback: return user-profile based recommendations
    logger.info("fallback recommendations generated")
    recommendations = []
    seen_ids = set(liked_recipe_ids) if liked_recipe_ids else set()

    for idx, score in zip(user_top_idx, user_top_scores):
        recipe_id = idx_to_recipe_id[idx]
        if recipe_id not in seen_ids:
            reason = _generate_reason(
                recipe_id=recipe_id,
                recipe_db=recipe_db,
                user_prefs=user_prefs,
                liked_recipe_name=None,
                similarity_score=0.0,
                is_liked_based=False
            )
            recommendations.append({
                "recipe_id": recipe_id,
                "score": float(score),
                "reason": reason
            })
            if len(recommendations) >= top_k:
                break

    return recommendations
