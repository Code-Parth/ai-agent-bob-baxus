import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------
# Data Loading & Feature Preparation
# ------------------------------------


@st.cache_data
def load_dataset(csv_path='501-Bottle-Dataset.csv'):
    df = pd.read_csv(csv_path)
    df['avg_msrp'] = pd.to_numeric(df['avg_msrp'], errors='coerce').fillna(0)
    df['abv'] = pd.to_numeric(df['abv'], errors='coerce').fillna(0)
    df['proof'] = pd.to_numeric(
        df['proof'], errors='coerce').fillna(df['abv'] * 2)
    return df


@st.cache_data
def prepare_features(df: pd.DataFrame):
    cat_cols = ['spirit_type']
    num_cols = ['avg_msrp', 'abv', 'proof']
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_feats = enc.fit_transform(df[cat_cols].fillna('Unknown'))
    scaler = MinMaxScaler()
    num_feats = scaler.fit_transform(df[num_cols])
    feats = np.hstack([cat_feats, num_feats])
    id_to_idx = {bid: idx for idx, bid in enumerate(df['id'])}
    return feats, id_to_idx

# ------------------------------------
# Recommendation Logic
# ------------------------------------


@st.cache_data
def compute_recommendations(df, features, id_to_idx, user_ids: list):
    mask = df['id'].isin(user_ids)
    if mask.sum() == 0:
        return [], {}
    profile_vec = features[mask].mean(axis=0)
    sims = cosine_similarity([profile_vec], features).flatten()
    df['similarity_overall'] = sims
    general = (
        df[~df['id'].isin(user_ids)]
        .nlargest(5, 'similarity_overall')
          [['id', 'name', 'spirit_type', 'avg_msrp', 'similarity_overall']]
        .rename(columns={'name': 'bottle', 'spirit_type': 'spirit_type', 'avg_msrp': 'msrp', 'similarity_overall': 'score'})
        .to_dict(orient='records')
    )
    by_bottle = {}
    for bid in user_ids:
        idx = id_to_idx.get(bid)
        if idx is None:
            continue
        sims_b = cosine_similarity([features[idx]], features).flatten()
        df['sim_temp'] = sims_b
        recs = (
            df[~df['id'].isin(user_ids)]
            .nlargest(5, 'sim_temp')
              [['id', 'name', 'spirit_type', 'avg_msrp', 'sim_temp']]
            .rename(columns={'name': 'bottle', 'spirit_type': 'spirit_type', 'avg_msrp': 'msrp', 'sim_temp': 'score'})
            .to_dict(orient='records')
        )
        by_bottle[bid] = recs
    return general, by_bottle


# ------------------------------------
# Fetch User Bar
# ------------------------------------
API_BASE = "https://services.baxus.co/api/bar/user/"


def fetch_user_bar(username: str):
    resp = requests.get(f"{API_BASE}{username}")
    resp.raise_for_status()
    return resp.json()

# ------------------------------------
# Streamlit API Endpoint via Query Param
# ------------------------------------


def serve_api():
    # Read query parameters directly
    qp = st.query_params
    username = qp.get("username") or qp.get("user") or None

    if username:
        try:
            user_bar = fetch_user_bar(username)
        except Exception as e:
            st.error(f"Error fetching user bar for '{username}': {e}")
            st.stop()

        user_ids = [item['product']['id'] for item in user_bar]
        general, by_bottle = compute_recommendations(
            df, features, id_to_idx, user_ids)

        # Output JSON and halt the rest of the UI
        st.json({"general": general, "by_bottle": by_bottle})
        st.stop()


# ------------------------------------
# Main App UI
# ------------------------------------
if __name__ == '__main__':
    df = load_dataset()
    features, id_to_idx = prepare_features(df)

    # Serve API if username query param is present
    serve_api()

    st.title("Bob: Whisky Recommendation Agent")
    username = st.text_input("Enter your Baxus username:")

    if username:
        try:
            user_bar = fetch_user_bar(username)
        except Exception as e:
            st.error(f"Error fetching data for '{username}': {e}")
            st.stop()

        if not user_bar:
            st.warning("No bottles found in your bar.")
        else:
            bar_records = []
            for item in user_bar:
                prod = item.get('product', {})
                bar_records.append({
                    'id': prod.get('id'),
                    'Bottle': prod.get('name', 'Unknown'),
                    'Spirit': prod.get('spirit', 'Unknown'),
                    'Price (MSRP)': prod.get('average_msrp', 0),
                    'Proof': prod.get('proof', None)
                })
            bar_df = pd.DataFrame(bar_records)
            st.subheader("Your Bar Overview")
            st.dataframe(bar_df[['Bottle', 'Spirit', 'Price (MSRP)', 'Proof']])

            user_ids = bar_df['id'].tolist()
            general, by_bottle = compute_recommendations(
                df, features, id_to_idx, user_ids)

            st.subheader("General Recommendations")
            gen_df = pd.DataFrame(general).rename(columns={
                'bottle': 'Bottle', 'spirit_type': 'Spirit Type', 'msrp': 'Price (MSRP)', 'score': 'Score'})
            st.dataframe(
                gen_df[['Bottle', 'Spirit Type', 'Price (MSRP)', 'Score']])

            st.subheader("Recommendations Based on Bottle")
            for bid, recs in by_bottle.items():
                bottle_name = bar_df.loc[bar_df['id'] == bid, 'Bottle'].iloc[0]
                with st.expander(f"Top 5 similar to {bottle_name}"):
                    rec_df = pd.DataFrame(recs).rename(columns={
                        'bottle': 'Bottle', 'spirit_type': 'Spirit Type', 'msrp': 'Price (MSRP)', 'score': 'Score'})
                    st.table(
                        rec_df[['Bottle', 'Spirit Type', 'Price (MSRP)', 'Score']])

            st.markdown(
                "**How it works:** We compute an overall taste profile vector and per-bottle vectors, then use cosine similarity to surface 5 top matches in each case."
            )
