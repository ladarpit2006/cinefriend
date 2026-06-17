import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# Create models_data directory
os.makedirs("models_data", exist_ok=True)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# 1. LOAD DATASETS
# -------------------------------------------------------------------------
print("Loading datasets...")
ratings_df = pd.read_csv("Dataset/rating.csv")
ratings_subset = ratings_df.iloc[:100000].copy()

movies_metadata = pd.read_csv("DataSet.csv")
# Clean and prepare features for SentenceTransformers
movies_metadata["tag"] = movies_metadata["tag"].fillna("")
movies_metadata["genres"] = movies_metadata["genres"].fillna("")
movies_metadata["allFeatures"] = movies_metadata["genres"] + " " + movies_metadata["tag"]

# -------------------------------------------------------------------------
# 2. TRAIN & SAVE AUTOENCODER MODEL
# -------------------------------------------------------------------------
print("Training Autoencoder model...")

# Pivot table user-movie matrix
user_item = ratings_subset.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

ae_user_ids = user_item.index.tolist()
ae_movie_ids = user_item.columns.tolist()

matrix_data = user_item.values
matrix_tensor = torch.tensor(matrix_data, dtype=torch.float32).to(device)
num_users, num_movies_ae = matrix_tensor.shape

class AutoEncoder(nn.Module):
    def __init__(self, num_movies):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_movies, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, num_movies)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

ae_model = AutoEncoder(num_movies_ae).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)

# Training loop
epochs = 15
ae_model.train()
for epoch in range(epochs):
    output = ae_model(matrix_tensor)
    loss = loss_fn(output, matrix_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Autoencoder Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save Autoencoder state
torch.save(ae_model.state_dict(), "models_data/autoencoder_model.pth")
with open("models_data/autoencoder_mappings.pkl", "wb") as f:
    pickle.dump({
        "user_ids": ae_user_ids,
        "movie_ids": ae_movie_ids,
        "ratings_matrix": matrix_data
    }, f)

print("Autoencoder model and mappings saved.")

# -------------------------------------------------------------------------
# 3. TRAIN & SAVE NEURAL COLLABORATIVE FILTERING (NCF) MODEL
# -------------------------------------------------------------------------
print("Training NCF model...")

ncf_user_ids = ratings_subset["userId"].unique()
ncf_movie_ids = ratings_subset["movieId"].unique()

user_to_idx = {u: i for i, u in enumerate(ncf_user_ids)}
movie_to_idx = {m: i for i, m in enumerate(ncf_movie_ids)}

ratings_subset["user_idx"] = ratings_subset["userId"].map(user_to_idx)
ratings_subset["movie_idx"] = ratings_subset["movieId"].map(movie_to_idx)

class NCF(nn.Module):
    def __init__(self, num_users, num_movies, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.movie_emb = nn.Embedding(num_movies, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user, movie):
        u = self.user_emb(user)
        m = self.movie_emb(movie)
        x = torch.cat([u, m], dim=1)
        return self.mlp(x).squeeze()

num_users_ncf = len(ncf_user_ids)
num_movies_ncf = len(ncf_movie_ids)
ncf_model = NCF(num_users_ncf, num_movies_ncf).to(device)

ncf_loss_fn = nn.MSELoss()
ncf_optimizer = torch.optim.Adam(ncf_model.parameters(), lr=0.001)

# PyTorch DataLoader for training
users_t = torch.tensor(ratings_subset["user_idx"].values, dtype=torch.long).to(device)
movies_t = torch.tensor(ratings_subset["movie_idx"].values, dtype=torch.long).to(device)
ratings_t = torch.tensor(ratings_subset["rating"].values, dtype=torch.float32).to(device)

epochs_ncf = 15
ncf_model.train()
for epoch in range(epochs_ncf):
    predictions = ncf_model(users_t, movies_t)
    loss = ncf_loss_fn(predictions, ratings_t)
    ncf_optimizer.zero_grad()
    loss.backward()
    ncf_optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"NCF Epoch {epoch+1}/{epochs_ncf}, Loss: {loss.item():.4f}")

# Save NCF state
torch.save(ncf_model.state_dict(), "models_data/ncf_model.pth")
with open("models_data/ncf_mappings.pkl", "wb") as f:
    pickle.dump({
        "user_ids": ncf_user_ids.tolist(),
        "movie_ids": ncf_movie_ids.tolist(),
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx
    }, f)

print("NCF model and mappings saved.")

# Clean up GPU memory
import gc
try:
    del ae_model
    del ncf_model
    del matrix_tensor
    del users_t
    del movies_t
    del ratings_t
except NameError:
    pass
gc.collect()
torch.cuda.empty_cache()

# -------------------------------------------------------------------------
# 4. PRECOMPUTE SENTENCE TRANSFORMER EMBEDDINGS
# -------------------------------------------------------------------------
print("Generating Sentence Transformer embeddings (all-mpnet-base-v2)...")
transformer = SentenceTransformer("all-mpnet-base-v2", device=str(device))
features_list = movies_metadata["allFeatures"].tolist()

# Compute embeddings with smaller batch size to avoid GPU memory OOM
embeddings = transformer.encode(features_list, show_progress_bar=True, batch_size=32)
np.save("models_data/movie_embeddings.npy", embeddings)
print(f"Embeddings saved. Shape: {embeddings.shape}")

print("All models and embeddings trained and prepared successfully!")

