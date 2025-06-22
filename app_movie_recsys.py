import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load all saved data

with open("user_item_matrix.pkl", "rb") as f:
    user_item_filled = pickle.load(f)
with open("user_sim_cosine.pkl", "rb") as f:
    user_sim_cosine = pickle.load(f)
with open("user_sim_pearson.pkl", "rb") as f:
    user_sim_pearson = pickle.load(f)
with open("user_item_centered.pkl", "rb") as f:
    user_item_centered = pickle.load(f)
with open("user_means.pkl", "rb") as f:
    train_user_means = pickle.load(f)
with open("item_similarity_cosine.pkl", "rb") as f:
    item_similarity_cosine = pickle.load(f)
with open("item_similarity_pearson.pkl", "rb") as f:
    item_similarity_pearson = pickle.load(f)
with open("item_means.pkl", "rb") as f:
    item_means = pickle.load(f)
with open("movies_df.pkl", "rb") as f:
    movies_df = pickle.load(f)
with open("posters.json", "r") as f:
    posters_dict = json.load(f)



# Prediction functions
def get_top_k_similar(similarities, k=30, min_similarity=0.0):
    similarities = similarities[similarities >= min_similarity]
    return similarities.sort_values(ascending=False).head(k)

def predict_rating_user_cosine(user_id, movie_id, k=30, min_similarity=0.0):
    if user_id not in user_sim_cosine.index or movie_id not in user_item_filled.columns:
        return None
    sims = get_top_k_similar(user_sim_cosine.loc[user_id], k, min_similarity)
    neighbors = sims.index
    ratings = user_item_filled.loc[neighbors, movie_id]
    mask = ratings > 0
    sims = sims[mask]
    ratings = ratings[mask]
    if len(ratings) == 0 or sims.abs().sum() == 0:
        return None
    return np.dot(sims, ratings) / sims.abs().sum()

def predict_rating_user_pearson(user_id, movie_id, k=30, min_similarity=0.0):
    if user_id not in user_sim_pearson.index or movie_id not in user_item_centered.columns:
        return None
    sims = get_top_k_similar(user_sim_pearson.loc[user_id], k, min_similarity)
    neighbors = sims.index
    ratings = user_item_centered.loc[neighbors, movie_id]
    mask = ratings.notna()
    sims = sims[mask]
    ratings = ratings[mask]
    if len(ratings) == 0 or sims.abs().sum() == 0:
        return None
    return train_user_means[user_id] + np.dot(sims, ratings) / sims.abs().sum()

def predict_rating_item_cosine(user_id, movie_id, k=30, min_similarity=0.0):
    if movie_id not in item_similarity_cosine.index or user_id not in user_item_filled.index:
        return None
    sims = get_top_k_similar(item_similarity_cosine.loc[movie_id], k, min_similarity)
    neighbors = sims.index
    ratings = user_item_filled.loc[user_id, neighbors]
    mask = ratings > 0
    sims = sims[mask]
    ratings = ratings[mask]
    if len(ratings) == 0 or sims.abs().sum() == 0:
        return None
    return np.dot(sims, ratings) / sims.abs().sum()

def predict_rating_item_pearson(user_id, movie_id, k=30, min_similarity=0.0):
    if movie_id not in item_similarity_pearson.index or user_id not in user_item_centered.index:
        return None
    sims = get_top_k_similar(item_similarity_pearson.loc[movie_id], k, min_similarity)
    neighbors = sims.index
    ratings = user_item_centered.loc[user_id, neighbors]
    mask = ratings.notna()
    sims = sims[mask]
    ratings = ratings[mask]
    if len(ratings) == 0 or sims.abs().sum() == 0:
        return None
    return item_means[movie_id] + np.dot(sims, ratings) / sims.abs().sum()



#=== Streamlit App ===
st.set_page_config(layout="wide")
st.title("Movie Recommender System (MovieLens 100K)")

user_id = st.number_input("Enter User ID (1–943):", min_value=1, max_value=943, value=1)

movie_title = st.selectbox("Select a movie:", movies_df["title"].values)
movie_id = movies_df[movies_df["title"] == movie_title]["movie_id"].values[0]

model_type = st.selectbox("Choose a Collaborative Filtering model:", [
    "User-User Cosine", "User-User Pearson",
    "Item-Item Cosine", "Item-Item Pearson"
])

k = st.slider("Top-K Neighbors:", min_value=1, max_value=50, value=30)
min_sim = st.slider("Minimum Similarity Threshold:", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

if st.button("Predict Rating"):
    if model_type == "User-User Cosine":
        pred = predict_rating_user_cosine(user_id, movie_id, k, min_sim)
    elif model_type == "User-User Pearson":
        pred = predict_rating_user_pearson(user_id, movie_id, k, min_sim)
    elif model_type == "Item-Item Cosine":
        pred = predict_rating_item_cosine(user_id, movie_id, k, min_sim)
    elif model_type == "Item-Item Pearson":
        pred = predict_rating_item_pearson(user_id, movie_id, k, min_sim)
    else:
        pred = None

    if pred is not None:
        st.success(f"Predicted Rating for '{movie_title}' (User {user_id}): {pred:.2f}")
    else:
        st.warning("Unable to predict rating. Try adjusting K or similarity threshold.")