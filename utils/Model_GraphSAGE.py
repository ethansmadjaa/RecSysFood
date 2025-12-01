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
from sklearn.preprocessing import LabelEncoder


def pre_processing(user_preference_db, recipe_db, interaction_db):
    # Recipe processing
    recipe_features = [
        "RecipeId",
        "Calories",
        "FatContent",
        "ProteinContent",
        "SugarContent",
        "PrepTime_min",
        "TotalTime_min",
        "Is_Vegan",
        "Is_Vegetarian",
        "Contains_Pork",
        "Contains_Alcohol",
        "Contains_Gluten",
        "Contains_Nuts",
        "Contains_Dairy",
        "Contains_Egg",
        "Contains_Fish",
        "Contains_Soy",
        "Is_Breakfast_Brunch",
        "Is_Dessert",
    ]

    recipe_db = recipe_db[recipe_features]  # Keep features for recipe nodes attribute

    # User preference processing
    for col in ["meal_type", "nutrition_goal_protein", "nutrition_goal_calories"]:
        le = LabelEncoder()
        user_preference_db[col] = le.fit_transform(user_preference_db[col].astype(str))

    # Interraction processing
    interaction_db = interaction_db.drop("interaction_id", axis=1)

    return user_preference_db, recipe_db, interaction_db


def BuildGraph(user_preference_db, recipe_db, interaction_db):
    G = nx.Graph()

    # Define users preference node
    for _, row in user_preference_db.iterrows():
        node_id = row["user_id"]  # colonne pour identifier le noeud
        attrs = row.drop("user_id").to_dict()  # node features
        attrs["node_type"] = "user"  # node type

        G.add_node(node_id, bipartite=0, **attrs)

    # Define recipe node
    for _, row in recipe_db.iterrows():
        node_id = row["RecipeId"]  # columns de identify node
        attrs = row.drop("RecipeId").to_dict()  # node features
        attrs["node_type"] = "recipe"  # node type

        G.add_node(node_id, bipartite=1, **attrs)

    # Define edge
    for _, row in interaction_db.iterrows():
        G.add_edge(f"u_{row['user_id']}", f"r_{row['recipe_id']}", rating=row["rating"])

        return G


def Graph_to_PyG(G, user_preference_db, recipe_db):
    # Map node IDs
    unique_u = user_preference_db["user_id"].unique()
    unique_r = recipe_db["RecipeId"].unique()

    use_dict = {f"u_{str(u)}": i for i, u in enumerate(unique_u)}
    recip_dict = {f"r_{str(r)}": i for i, r in enumerate(unique_r)}


    # Create features matrix
    users_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "user"]
    recipe_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "recipe"]

    num_recipe_features = [
        "Calories",
        "FatContent",
        "ProteinContent",
        "SugarContent",
        "PrepTime_min",
        "TotalTime_min",
        "Is_Vegan",
        "Is_Vegetarian",
        "Contains_Pork",
        "Contains_Alcohol",
        "Contains_Gluten",
        "Contains_Nuts",
        "Contains_Dairy",
        "Contains_Egg",
        "Contains_Fish",
        "Contains_Soy",
        "Is_Breakfast_Brunch",
        "Is_Dessert",
    ]

    # Create pyG data object
    data = HeteroData()

    # Convert Users nodes and features to pyG data object
    user_features = []

    for u in users_nodes:
        f = list(G.nodes[u].values())
        f = [v for v in f if isinstance(v, (int, float))]
        user_features.append(f)

    x_user = torch.tensor(user_features, dtype=torch.float)  # your 5 user features
    data["user"].x = x_user

    # Convert recipes nodes and features to pyG data object
    recipe_feature = []
    recipe_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "recipe"]

    for r in recipe_nodes:
        f = []
        for feat in num_recipe_features:
            f.append(G.nodes[r].get(feat, 0.0))
        recipe_feature.append(f)

    x_recipe = torch.tensor(recipe_feature, dtype=torch.float)
    data["recipe"].x = x_recipe

    # Convert edges to pyG data object
    edge_list = []

    for u, r, n in G.edges(data=True):
        if str(u).startswith("u_") and str(r).startswith("r_"):
            edge_list.append([use_dict[u], recip_dict[r]])

        elif str(u).startswith("r_") and str(r).startswith("u_"):
            edge_list.append([use_dict[r], recip_dict[u]])

        else:
            continue

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    data["user", "rating", "recipe"].edge_index = edge_index

    data = T.ToUndirected()(data)  # Add reverse data

    return data


# Implement the GNN Model
class GraphSAGELinkPredictor(torch.nn.Module):
    def __init__(self, user_dim, recipe_dim, hidden_channels, out_channels):
        super().__init__()

        #  Project both feature types -> hidden_dim
        self.user_lin = nn.Linear(user_dim, hidden_channels)
        self.recipe_lin = nn.Linear(recipe_dim, hidden_channels)

        # Define heterogeneous SAGEConv layers
        self.conv1 = HeteroConv(
            {
                ("user", "rating", "recipe"): SAGEConv(
                    hidden_channels, hidden_channels
                ),
                ("recipe", "rev_rating", "user"): SAGEConv(
                    hidden_channels, hidden_channels
                ),
            }
        )
        self.conv2 = HeteroConv(
            {
                ("user", "rating", "recipe"): SAGEConv(hidden_channels, out_channels),
                ("recipe", "rev_rating", "user"): SAGEConv(
                    hidden_channels, out_channels
                ),
            }
        )
        self.dropout = nn.Dropout(p=0.5)

    def Encoder(self, x_dict, edge_index_dict):
        # Get the Dense node features
        x_dict = {
            "user": self.user_lin(x_dict["user"]),
            "recipe": self.recipe_lin(x_dict["recipe"]),
        }

        # Define SAGEConv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: nn.functional.relu(v) for k, v in x_dict.items()}
        z = self.conv2(x_dict, edge_index_dict)
        z = {k: self.dropout(v) for k, v in z.items()}

        return z

    def Decoder(self, z, edge_label_index):
        # Select the type the source and destination nodes
        src, _, dst = ("user", "rating", "recipe")

        z_src = z[src][edge_label_index[0]]
        z_dest = z[dst][edge_label_index[1]]

        # Normalisation of the embedding
        z_src = nn.functional.normalize(z_src, p=2, dim=1)
        z_dest = nn.functional.normalize(z_dest, p=2, dim=1)

        # Calculate the dot product for each edge
        scores = (z_src * z_dest).sum(dim=1)

        return scores


# Define train function
def training(model, train_data, optimizer, loss):
    model.train()

    # Get the training data
    edge_idx = train_data["user", "rating", "recipe"].edge_index
    pos_edge_lab = train_data["user", "rating", "recipe"].pos_edge_label_index

    # Create negative sampling
    nbr_user = train_data["user"].num_nodes
    nbr_recipe = train_data["recipe"].num_nodes

    neg_edge_lab = negative_sampling(
        edge_index=edge_idx,
        num_nodes=(nbr_user, nbr_recipe),
        num_neg_samples=pos_edge_lab.size(1),
    )

    # Combine positive and negative sampling
    edge_label_idx = torch.cat([pos_edge_lab, neg_edge_lab], dim=1).long()

    # Comput embedding
    z = model.Encoder(
        x_dict={"user": train_data["user"].x, "recipe": train_data["recipe"].x},
        edge_index_dict={
            ("user", "rating", "recipe"): train_data[
                "user", "rating", "recipe"
            ].edge_index,
            ("recipe", "rev_rating", "user"): train_data[
                "recipe", "rev_rating", "user"
            ].edge_index,
        },
    )

    # Compute prediction
    scores = model.Decoder(z, edge_label_idx)

    # Create ground-truth edge label
    pos_labels = torch.ones(pos_edge_lab.size(1))  # positive edges → label = 1
    neg_labels = torch.zeros(neg_edge_lab.size(1))  # negative edges → label = 0
    edge_labels = torch.cat([pos_labels, neg_labels], dim=0)

    # Calculate loss
    loss_fct = loss(scores, edge_labels)
    optimizer.zero_grad()
    loss_fct.backward()
    optimizer.step()

    return loss_fct.item()


# Define test function
@torch.no_grad()
def test(model, val_test_data):
    model.eval()

    # Get the necessary input for encoding
    edge_idx = val_test_data["user", "rating", "recipe"].edge_index
    pos_edge_lab = val_test_data["user", "rating", "recipe"].pos_edge_label_index

    # Create neg sample manually
    nbr_user = val_test_data["user"].num_nodes
    nbr_recipe = val_test_data["recipe"].num_nodes

    neg_edge_lab = negative_sampling(
        edge_index=edge_idx,
        num_nodes=(nbr_user, nbr_recipe),
        num_neg_samples=pos_edge_lab.size(1),
    )

    edge_label_index = torch.cat([pos_edge_lab, neg_edge_lab], dim=1).long()

    # Generate nodes encoding
    z = model.Encoder(
        x_dict={"user": val_test_data["user"].x, "recipe": val_test_data["recipe"].x},
        edge_index_dict={
            ("user", "rating", "recipe"): val_test_data[
                "user", "rating", "recipe"
            ].edge_index,
            ("recipe", "rev_rating", "user"): val_test_data[
                "recipe", "rev_rating", "user"
            ].edge_index,
        },
    )

    # Get the prediction score
    scores = model.Decoder(z, edge_label_index)

    # Create ground-truth edge label
    pos_labels = torch.ones(pos_edge_lab.size(1))  # positive edges -> label = 1
    neg_labels = torch.zeros(neg_edge_lab.size(1))  # negative edges -> label = 0
    edge_labels = torch.cat([pos_labels, neg_labels], dim=0)

    # Juste avant roc_auc_score
    if edge_labels.numel() == 0:
        print("No edges to evaluate!")
        return 0.0, 0.0

    # compute auc score
    auc = roc_auc_score(edge_labels, scores)
    auc_pr = average_precision_score(edge_labels, scores)

    return auc, auc_pr


def Train_GNN(user_preference_db, recipe_db, interaction_db):
    # Process dataset
    user_preference_db, recipe_db, interaction_db = pre_processing(
        user_preference_db, recipe_db, interaction_db
    )

    # Build Biparte Graph using datasets
    G = BuildGraph(user_preference_db, recipe_db, interaction_db)

    # Convert the Graph into numerical data
    data = Graph_to_PyG(G, user_preference_db, recipe_db)

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

    # Initialization of the hyperparameter
    user_dim = train_data["user"].x.size(1)
    recipe_dim = train_data["recipe"].x.size(1)

    model = GraphSAGELinkPredictor(
        user_dim=user_dim, recipe_dim=recipe_dim, hidden_channels=64, out_channels=32
    )

    Adam_opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fct = nn.BCEWithLogitsLoss()

    epochs = 15
    loss = []
    for i in range(epochs):
        loss_value = training(model, train_data, optimizer=Adam_opt, loss=loss_fct)
        print(f"Epoch {i + 1}/{epochs} - Loss: {loss_value:.4f}")
        loss.append(loss_value)

    # Test the model
    auc, auc_pr = test(model, test_data)
    print("Final test AUC score :", auc)
    print("Final test NDCG@k score :", auc_pr)

    return data, model


# Get recommendation
def Recommendation(model, data, user_id, recipe_db, top_k=10):
    model.eval()

    with torch.no_grad():
        # Calculer embeddings via l'Encoder du modèle
        out = model.Encoder(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)

    # Embeddings
    recipe_emb = out["recipe"]  # [num_recipes, hidden_dim]

    # Utiliser le Decoder du modèle pour calculer les scores
    # On crée un edge_label_index fictif pour tous les recipes
    edge_label_index = torch.stack(
        [
            torch.full((recipe_emb.size(0),), user_id, dtype=torch.long),  # user index
            torch.arange(recipe_emb.size(0), dtype=torch.long),  # recipe indices
        ]
    )

    scores = model.Decoder(out, edge_label_index)

    # Recettes déjà notées
    u, r = data["user", "rating", "recipe"].edge_index
    already_rated = r[u == user_id].tolist()
    scores[already_rated] = -1e9

    # Top-k
    recommended = scores.topk(top_k).indices.tolist()

    # Get original IDs
    unique_r = recipe_db["RecipeId"].unique()
    recip_dict = {f"r_{str(r)}": i for i, r in enumerate(unique_r)}

    # Convert Graph recipe IDs to original recipe IDs
    idx_to_recipe = {v: k for k, v in recip_dict.items()}
    recommended_recipe = [idx_to_recipe[i].split("_")[1] for i in recommended]
    recommended_recipe = list(map(int, recommended_recipe))

    return recommended_recipe


# Load the dataset
user_preference_db = pd.read_csv(
    "C:/Users/Ilian/OneDrive/Documents/GitHub/RecSysFood/models/user_profil.txt"
)
recipe_db = pd.read_csv(
    "C:/Users/Ilian/OneDrive/Documents/GitHub/RecSysFood/models/recipes.csv"
)
interaction_db = pd.read_csv(
    "C:/Users/Ilian/OneDrive/Documents/GitHub/RecSysFood/models/interaction.txt"
)

# Train your model
data, model = Train_GNN(user_preference_db, recipe_db, interaction_db)

# Generate recommendation
recommended_recipe = Recommendation(model, data, 13, recipe_db, top_k=10)

for (
    _,
    r,
) in recipe_db.iterrows():
    if r["RecipeId"] in recommended_recipe:
        print(r[["RecipeId", "Name"]])
