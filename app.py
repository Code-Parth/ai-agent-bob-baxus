import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Caching dataset load


def load_dataset():
    df = pd.read_csv('501-Bottle-Dataset.csv')
    # Standardize numeric columns
    df['avg_msrp'] = pd.to_numeric(df['avg_msrp'], errors='coerce').fillna(0)
    df['abv'] = pd.to_numeric(df['abv'], errors='coerce').fillna(0)
    df['proof'] = pd.to_numeric(
        df['proof'], errors='coerce').fillna(df['abv'] * 2)
    return df

# Caching API fetch


def fetch_user_bar(username: str):
    url = f"https://services.baxus.co/api/bar/user/{username}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

# Prepare feature matrix


def prepare_features(df: pd.DataFrame):
    cat_cols = ['spirit_type']
    num_cols = ['avg_msrp', 'abv', 'proof']

    # OneHotEncoder now uses sparse_output instead of sparse
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_feats = enc.fit_transform(df[cat_cols].fillna('Unknown'))

    scaler = MinMaxScaler()
    num_feats = scaler.fit_transform(df[num_cols])

    # Combine categorical and numerical features
    features = np.hstack([cat_feats, num_feats])
    return features, enc, scaler

# Build user taste profile


def get_user_profile(user_data: list, df: pd.DataFrame, features: np.ndarray):
    user_ids = [item['product']['id'] for item in user_data]
    mask = df['id'].isin(user_ids)
    if not mask.any():
        return None, user_ids
    profile_vec = features[mask].mean(axis=0)
    return profile_vec, user_ids

# Recommendation function


def recommend(profile_vec: np.ndarray, df: pd.DataFrame, features: np.ndarray, user_ids: list, top_n: int = 10):
    sims = cosine_similarity([profile_vec], features)[0]
    df_rec = df.copy()
    df_rec['similarity'] = sims
    df_rec = df_rec[~df_rec['id'].isin(user_ids)]
    return df_rec.sort_values('similarity', ascending=False).head(top_n)


# Streamlit App
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
        st.subheader("Your Bar")
        bar_records = []
        for item in user_bar:
            prod = item.get('product', {})
            bar_records.append({
                'Name': prod.get('name', 'Unknown'),
                'Brand': prod.get('brand', 'Unknown'),
                'Spirit': prod.get('spirit', 'Unknown'),
                'Price (MSRP)': prod.get('average_msrp', 0),
                'Proof': prod.get('proof', None)
            })
        bar_df = pd.DataFrame(bar_records)
        st.dataframe(bar_df)

        # Load data and features
        df = load_dataset()
        features, enc, scaler = prepare_features(df)
        profile_vec, user_ids = get_user_profile(user_bar, df, features)

        if profile_vec is None:
            st.warning("Could not build taste profile from your bottles.")
        else:
            recs = recommend(profile_vec, df, features, user_ids)
            st.subheader("Personalized Recommendations")
            rec_display = recs[['name', 'spirit_type',
                                'avg_msrp', 'similarity']]
            rec_display = rec_display.rename(columns={
                'name': 'Bottle',
                'spirit_type': 'Spirit Type',
                'avg_msrp': 'Price (MSRP)',
                'similarity': 'Match Score'
            })
            st.dataframe(rec_display)
            st.markdown(
                "**How it works:** We encode spirit type, price, ABV, and proof and use cosine similarity to rank bottles matching your bar profile.")
