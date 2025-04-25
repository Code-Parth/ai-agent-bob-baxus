import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Dataset loader


def load_dataset():
    df = pd.read_csv('501-Bottle-Dataset.csv')
    df['avg_msrp'] = pd.to_numeric(df['avg_msrp'], errors='coerce').fillna(0)
    df['abv'] = pd.to_numeric(df['abv'], errors='coerce').fillna(0)
    df['proof'] = pd.to_numeric(
        df['proof'], errors='coerce').fillna(df['abv'] * 2)
    return df

# Fetch user bar from API


def fetch_user_bar(username: str):
    url = f"https://services.baxus.co/api/bar/user/{username}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

# Prepare features


def prepare_features(df: pd.DataFrame):
    cat_cols = ['spirit_type']
    num_cols = ['avg_msrp', 'abv', 'proof']
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_feats = enc.fit_transform(df[cat_cols].fillna('Unknown'))
    scaler = MinMaxScaler()
    num_feats = scaler.fit_transform(df[num_cols])
    features = np.hstack([cat_feats, num_feats])
    # Map bottle id to index in features array
    id_to_idx = {bid: idx for idx, bid in enumerate(df['id'])}
    return features, id_to_idx


# Streamlit App Interface
st.title("Bob: Whisky Recommendation Agent")
username = st.text_input("Enter your Baxus username:")

if username:
    # Fetch user bar
    try:
        user_bar = fetch_user_bar(username)
    except Exception as e:
        st.error(f"Error fetching data for '{username}': {e}")
        st.stop()

    if not user_bar:
        st.warning("No bottles found in your bar.")
    else:
        # Display user's bar overview
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

        # Load dataset and features
        df = load_dataset()
        features, id_to_idx = prepare_features(df)

        # Build overall taste profile
        user_ids = bar_df['id'].tolist()
        user_mask = df['id'].isin(user_ids)
        profile_vec = features[user_mask].mean(axis=0)

        # General recommendations
        sims_overall = cosine_similarity([profile_vec], features)[0]
        df['similarity_overall'] = sims_overall
        gen_recs = df[~df['id'].isin(user_ids)].sort_values(
            'similarity_overall', ascending=False).head(5)
        st.subheader("General Recommendations")
        st.dataframe(
            gen_recs[['name', 'spirit_type', 'avg_msrp', 'similarity_overall']]
            .rename(columns={
                'name': 'Bottle', 'spirit_type': 'Spirit Type',
                'avg_msrp': 'Price (MSRP)', 'similarity_overall': 'Score'}))

        # Per-bottle recommendations using expanders
        st.subheader("Recommendations Based on Bottle")
        for _, row in bar_df.iterrows():
            bid = row['id']
            bname = row['Bottle']
            idx = id_to_idx.get(bid)
            if idx is None:
                continue
            sims = cosine_similarity([features[idx]], features)[0]
            df['sim_temp'] = sims
            # Exclude all owned bottles
            recs = df[~df['id'].isin(user_ids)].sort_values(
                'sim_temp', ascending=False).head(5)
            with st.expander(f"Top 5 similar to {bname}"):
                display = recs[['name', 'spirit_type', 'avg_msrp', 'sim_temp']]
                display = display.rename(columns={
                    'name': 'Bottle', 'spirit_type': 'Spirit Type',
                    'avg_msrp': 'Price (MSRP)', 'sim_temp': 'Score'
                })
                st.table(display)

        st.markdown(
            "**How it works:** We compute an overall taste profile vector and per-bottle vectors, then use cosine similarity to surface 5 top matches in each case.")
