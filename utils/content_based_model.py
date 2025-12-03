"""Content-based recommendation model using GraphSAGE for recipe embeddings.

This module implements a hybrid recommendation system that:
1. Creates recipe embeddings using a GraphSAGE autoencoder on recipe-recipe similarity graph
2. Creates user embeddings using SentenceTransformer on user preferences
3. Recommends recipes based on user profile similarity and liked recipe similarity
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
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from utils.logger import logger
# Download NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# Model save paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "trained"
MODEL_PATH = MODEL_DIR / "content_model.pt"
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


class GraphSAGEAutoencoder(nn.Module):
    """GraphSAGE-based autoencoder for learning recipe embeddings."""

    def __init__(self, input_dim: int, hidden_channels: int = 64, out_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.decoder = nn.Linear(out_channels, input_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode recipe features into embeddings."""
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Forward pass: encode then decode."""
        z = self.encode(x, edge_index)
        x_hat = self.decoder(z)
        return z, x_hat


def _create_recipe_interaction_graph(recipe_df: pd.DataFrame, n_neighbors: int = 10) -> pd.DataFrame:
    """Build recipe-recipe interaction graph based on feature similarity.

    Args:
        recipe_df: DataFrame with recipe features
        n_neighbors: Number of nearest neighbors per recipe

    Returns:
        DataFrame with columns [recipe_idx_1, recipe_idx_2] representing edges
    """
    # Create recipe index
    recipe_df = recipe_df.copy()
    recipe_df["recipe_idx"] = range(len(recipe_df))

    # Get numeric features for KNN
    available_features = [f for f in RECIPE_FEATURES if f in recipe_df.columns]
    feature_matrix = recipe_df[available_features].fillna(0).values

    # Build KNN graph
    knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(recipe_df)), metric="cosine")
    knn.fit(feature_matrix)
    _ , indices = knn.kneighbors(feature_matrix)

    # Create edge list (skip self-connections)
    interactions = []
    for i, neighbors in enumerate(indices):
        for j, neighbor_idx in enumerate(neighbors):
            if j > 0:  # Skip self
                interactions.append([i, neighbor_idx])

    return pd.DataFrame(interactions, columns=["recipe_idx_1", "recipe_idx_2"])


def _build_networkx_graph(recipe_df: pd.DataFrame, interaction_df: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from recipe data and interactions."""
    G = nx.Graph()

    available_features = [f for f in RECIPE_FEATURES if f in recipe_df.columns]

    # Add recipe nodes with features
    for idx, row in recipe_df.iterrows():
        attrs = {feat: float(row.get(feat, 0.0)) for feat in available_features}
        attrs["node_type"] = "recipe"
        G.add_node(idx, **attrs)

    # Add edges
    for _, row in interaction_df.iterrows():
        G.add_edge(row["recipe_idx_1"], row["recipe_idx_2"])

    return G


def _graph_to_pyg(G: nx.Graph) -> Data:
    """Convert NetworkX graph to PyTorch Geometric Data object."""
    recipe_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "recipe"]
    node_id_to_idx = {node_id: i for i, node_id in enumerate(recipe_nodes)}

    available_features = [f for f in RECIPE_FEATURES if f in G.nodes[recipe_nodes[0]]]

    # Build feature matrix
    features = []
    for node_id in recipe_nodes:
        attrs = G.nodes[node_id]
        f = [float(attrs.get(feat, 0.0)) for feat in available_features]
        features.append(f)

    x = torch.tensor(features, dtype=torch.float)

    # Build edge index
    edge_list = []
    for u, v in G.edges():
        if u in node_id_to_idx and v in node_id_to_idx:
            edge_list.append([node_id_to_idx[u], node_id_to_idx[v]])
            edge_list.append([node_id_to_idx[v], node_id_to_idx[u]])  # Undirected

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def _train_model(data: Data, model: GraphSAGEAutoencoder, epochs: int = 50, lr: float = 0.0005, verbose: bool = False):
    """Train the GraphSAGE autoencoder."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        z, x_pred = model(data.x, data.edge_index)
        loss = nn.functional.mse_loss(x_pred, data.x)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    return model


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
    """Create user embedding from preferences using SentenceTransformer.

    Args:
        user_prefs: Dict with keys like meal_types, calorie_goal, protein_goal
        sentence_model: Pre-loaded SentenceTransformer model

    Returns:
        User embedding tensor
    """
    # Build text from user preferences
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


def _calculate_similarity(query_embed: torch.Tensor, target_embeds: torch.Tensor, k: int = 100):
    """Calculate cosine similarity and return top-k matches.

    Args:
        query_embed: Query embedding (1D tensor)
        target_embeds: Target embeddings (2D tensor)
        k: Number of top matches to return

    Returns:
        Tuple of (indices, scores)
    """
    # Ensure both tensors are on CPU for consistency
    query_embed = query_embed.clone().detach().float().cpu()
    target_embeds = target_embeds.cpu()

    # Project to same dimension if needed
    if query_embed.dim() == 1:
        target_dim = target_embeds.size(1)
        if query_embed.size(0) != target_dim:
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


# =============================================================================
# PUBLIC API - These functions match the interface expected by the routes
# =============================================================================

def Train_GNN(
    user_preference_db: pd.DataFrame,
    recipe_db: pd.DataFrame,
    interaction_db: pd.DataFrame,
    epochs: int = 50
):
    """Train the content-based recommendation model.

    Args:
        user_preference_db: DataFrame with user preferences (kept for API compatibility)
        recipe_db: DataFrame with recipe features
        interaction_db: DataFrame with user-recipe interactions
        epochs: Number of training epochs

    Returns:
        Tuple of (recipe_embeddings, model, recipe_id_to_idx, idx_to_recipe_id, None)
    """
    print(f"Training content-based model on {len(recipe_db)} recipes...")

    # Normalize column names to lowercase
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
        return None, None, {}, {}, None

    # Initialize and train model
    input_dim = data.x.size(1)
    model = GraphSAGEAutoencoder(input_dim=input_dim, hidden_channels=64, out_channels=64)
    print("model initialized")
    if data.edge_index.size(1) > 0:
        model = _train_model(data, model, epochs=epochs, verbose=True)
        print("model trained")
    # Generate embeddings
    model.eval()
    with torch.no_grad():
        recipe_embeddings = model.encode(data.x, data.edge_index)
        print("recipe_embeddings generated")
    # Create mappings
    recipe_ids = recipe_db["recipeid"].tolist()
    recipe_id_to_idx = {rid: idx for idx, rid in enumerate(recipe_ids)}
    idx_to_recipe_id = {idx: rid for rid, idx in recipe_id_to_idx.items()}
    print("mappings created")   
    print(f"Generated {recipe_embeddings.size(0)} recipe embeddings of dimension {recipe_embeddings.size(1)}")
    print("recipe_embeddings, model, recipe_id_to_idx, idx_to_recipe_id, data returned")
    return recipe_embeddings, model, recipe_id_to_idx, idx_to_recipe_id, data


def save_model(
    model: GraphSAGEAutoencoder,
    data: Data,
    recipe_db: pd.DataFrame,
    recipe_id_to_idx: dict,
    idx_to_recipe_id: dict,
    recipe_scaler=None  # Kept for API compatibility
):
    """Save trained model and embeddings to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), MODEL_PATH)

    # Generate and save embeddings
    model.eval()
    with torch.no_grad():
        recipe_embeddings = model.encode(data.x, data.edge_index)
    torch.save(recipe_embeddings, RECIPE_EMBEDDINGS_PATH)

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
        Tuple of (model, recipe_embeddings, recipe_id_to_idx, idx_to_recipe_id, data, recipe_db) or None
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

    # Reconstruct model
    input_dim = data.x.size(1)
    model = GraphSAGEAutoencoder(input_dim=input_dim, hidden_channels=64, out_channels=64)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    print(f"Model loaded from {MODEL_PATH}")
    return model, recipe_embeddings, recipe_id_to_idx, idx_to_recipe_id, data, recipe_db


def model_exists() -> bool:
    """Check if a trained model exists."""
    return all(p.exists() for p in [MODEL_PATH, RECIPE_EMBEDDINGS_PATH, RECIPE_INTERACTION_PATH, RECIPE_MAPPING_PATH])


def user_exists_in_model(user_id: str) -> bool:
    """Check if we can generate recommendations for a user.

    For the content-based model, we can always generate recommendations
    as long as the model exists and the user has preferences.
    """
    return model_exists()


def _find_most_similar_liked_recipe(
    recipe_idx: int,
    liked_indices: list[int],
    recipe_embeddings: torch.Tensor,
    idx_to_recipe_id: dict,
    recipe_db: pd.DataFrame
) -> tuple[Optional[str], float]:
    """Find which liked recipe is most similar to a recommended recipe.

    Args:
        recipe_idx: Index of the recommended recipe
        liked_indices: List of indices of liked recipes
        recipe_embeddings: All recipe embeddings
        idx_to_recipe_id: Mapping from index to recipe ID
        recipe_db: DataFrame with recipe data

    Returns:
        Tuple of (liked_recipe_name, similarity_score) or (None, 0.0)
    """
    if not liked_indices:
        return None, 0.0

    rec_emb = recipe_embeddings[recipe_idx].unsqueeze(0)
    liked_embeds = recipe_embeddings[liked_indices]

    # Calculate cosine similarity with all liked recipes
    rec_norm = rec_emb / (rec_emb.norm() + 1e-8)
    liked_norm = liked_embeds / (liked_embeds.norm(dim=1, keepdim=True) + 1e-8)
    similarities = (liked_norm @ rec_norm.T).squeeze()

    if similarities.dim() == 0:
        best_idx = 0
        best_score = float(similarities)
    else:
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

    # Get the name of the most similar liked recipe
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
    """Generate a human-readable reason for recommending a recipe.

    Args:
        recipe_id: The recommended recipe ID
        recipe_db: DataFrame with recipe data
        user_prefs: User preferences dict
        liked_recipe_name: Name of the most similar liked recipe
        similarity_score: Similarity score to the liked recipe (0-1)
        is_liked_based: Whether this recommendation is based on a liked recipe

    Returns:
        A reason string explaining the recommendation
    """
    reasons = []

    # Get recipe data
    recipe_row = recipe_db[recipe_db["recipeid"] == recipe_id]
    if recipe_row.empty:
        return "Recommended for you"

    recipe = recipe_row.iloc[0]

    # 1. Similarity to liked recipe - only show if similarity is high enough
    if is_liked_based and liked_recipe_name and similarity_score > 0.3:
        reasons.append(f'Similar to "{liked_recipe_name}"')

    # 2. Recipe-specific attributes (most distinctive features)
    # Check for notable attributes that make this recipe unique

    # Quick recipes
    total_time = recipe.get("totaltime_min")
    if total_time and total_time <= 15:
        reasons.append("Ready in under 15 min")
    elif total_time and total_time <= 30:
        reasons.append("Quick to prepare")

    # Dietary attributes
    if recipe.get("is_vegan"):
        reasons.append("Vegan")
    elif recipe.get("is_vegetarian"):
        reasons.append("Vegetarian")

    # Nutritional highlights
    calories = recipe.get("calories")
    if calories and calories < 200:
        reasons.append("Low calorie")
    elif calories and calories > 600:
        reasons.append("Hearty meal")

    protein = recipe.get("proteincontent")
    if protein and protein > 30:
        reasons.append("High protein")

    # Meal type
    if recipe.get("is_breakfast_brunch"):
        reasons.append("Breakfast option")
    if recipe.get("is_dessert"):
        reasons.append("Sweet treat")

    # 3. Match with user preferences (secondary)
    meal_types = user_prefs.get("meal_types", [])
    if meal_types and not reasons:
        if recipe.get("is_breakfast_brunch") and "breakfast" in [m.lower() for m in meal_types]:
            reasons.append("Matches your breakfast preference")
        if recipe.get("is_dessert") and "dessert" in [m.lower() for m in meal_types]:
            reasons.append("Matches your dessert preference")

    # 4. Calorie/protein goals only if no other reasons
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

    # Return combined reasons or default
    if reasons:
        # Prioritize diverse reasons - similarity first, then attributes
        return " · ".join(reasons[:2])
    return "Recommended for you"


def get_recommendations_for_user(
    user_id: str,
    top_k: int = 15,
    user_prefs: Optional[dict] = None,
    liked_recipe_ids: Optional[list[int]] = None
) -> list[dict]:
    """Get recommendations for a user using the content-based model.

    Args:
        user_id: User UUID string
        top_k: Number of recommendations to return
        user_prefs: Optional dict with user preferences (meal_types, calorie_goal, protein_goal)
        liked_recipe_ids: Optional list of recipe IDs the user has liked

    Returns:
        List of dicts with recipe_id, score, and reason
    """
    loaded = load_model()
    if loaded is None:
        return []

    model, recipe_embeddings, recipe_id_to_idx, idx_to_recipe_id, data, recipe_db = loaded

    # Load sentence transformer for user embedding
    sentence_model = SentenceTransformer("all-mpnet-base-v2")

    # Create user embedding from preferences
    if user_prefs is None:
        user_prefs = {}

    user_embed = _create_user_embedding(user_prefs, sentence_model)

    # Get initial user-recipe recommendations
    k_user = min(500, len(recipe_embeddings))
    user_top_idx, user_top_scores = _calculate_similarity(user_embed, recipe_embeddings, k=k_user)
    user_top_idx = user_top_idx.tolist()
    user_top_scores = [float(s) for s in user_top_scores]

    # If user has liked recipes, refine recommendations
    if liked_recipe_ids and len(liked_recipe_ids) > 0:
        # Find indices of liked recipes
        liked_indices = []
        liked_id_to_idx_map = {}
        for rid in liked_recipe_ids:
            if rid in recipe_id_to_idx:
                idx = recipe_id_to_idx[rid]
                liked_indices.append(idx)
                liked_id_to_idx_map[idx] = rid

        if liked_indices:
            # Find best matching liked recipe for ranking recommendations
            best_liked_idx = None
            best_score = -float("inf")

            for liked_idx in liked_indices:
                liked_emb = recipe_embeddings[liked_idx]
                candidate_embeds = recipe_embeddings[user_top_idx]
                _, scores = _calculate_similarity(liked_emb, candidate_embeds, k=top_k)

                mean_score = float(scores.mean())
                if mean_score > best_score:
                    best_score = mean_score
                    best_liked_idx = liked_idx

            if best_liked_idx is not None:
                # Get final recommendations based on best liked recipe for ranking
                liked_emb = recipe_embeddings[best_liked_idx]
                candidate_embeds = recipe_embeddings[user_top_idx]
                top_local_idx, top_scores = _calculate_similarity(liked_emb, candidate_embeds, k=top_k * 2)

                recommendations = []
                for local_idx, score in zip(top_local_idx.tolist(), top_scores.tolist()):
                    global_idx = user_top_idx[local_idx]
                    recipe_id = idx_to_recipe_id[global_idx]

                    # Skip already liked recipes
                    if recipe_id not in liked_recipe_ids:
                        # Find which liked recipe is MOST similar to THIS specific recommendation
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
                        recommendations.append({
                            "recipe_id": recipe_id,
                            "score": float(score),
                            "reason": reason
                        })

                        if len(recommendations) >= top_k:
                            break

                return recommendations[:top_k]

    logger.info("fallback recommendations generated")
    # Fallback: return user-profile based recommendations
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
