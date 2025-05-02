import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import pickle
import faiss
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# ==============================
# Configuration
# ==============================
CSV_PATH = '501-Bottle-Dataset.csv'
IMAGE_DIR = 'bottle_images'
INDEX_PATH = 'bottle_index.faiss'
IDS_PATH = 'bottle_ids.pkl'
API_BASE = "https://services.baxus.co/api/bar/user/"

# Set page configuration
st.set_page_config(
    page_title="Bob: Whisky Recommendation Agent",
    page_icon="🥃",
    layout="wide"
)

# ==============================
# Data Loading & Feature Preparation
# ==============================


@st.cache_data
def load_dataset(csv_path=CSV_PATH):
    """Load and preprocess the bottle dataset"""
    df = pd.read_csv(csv_path)
    df['avg_msrp'] = pd.to_numeric(df['avg_msrp'], errors='coerce').fillna(0)
    df['abv'] = pd.to_numeric(df['abv'], errors='coerce').fillna(0)
    df['proof'] = pd.to_numeric(
        df['proof'], errors='coerce').fillna(df['abv'] * 2)
    return df


@st.cache_data
def prepare_content_features(df: pd.DataFrame):
    """Prepare content-based features (categorical + numerical)"""
    # Select features
    cat_cols = ['spirit_type']
    num_cols = ['avg_msrp', 'abv', 'proof']

    # Handle categorical features
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_feats = enc.fit_transform(df[cat_cols].fillna('Unknown'))

    # Handle numerical features
    scaler = MinMaxScaler()
    num_feats = scaler.fit_transform(df[num_cols])

    # Combine features
    feats = np.hstack([cat_feats, num_feats])
    id_to_idx = {bid: idx for idx, bid in enumerate(df['id'])}

    return feats, id_to_idx, enc  # Also return encoder for spirit type analysis

# ==============================
# Image Feature Extraction & FAISS Index
# ==============================


@st.cache_resource
def load_feature_extractor():
    """Load and return the MobileNetV2 model for image feature extraction"""
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')


def extract_image_features(img_path, model):
    """Extract features from an image using MobileNetV2"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(np.expand_dims(x, axis=0))
        feats = model.predict(x, verbose=0)
        return feats[0].astype('float32')
    except Exception as e:
        st.warning(f"Error processing image {img_path}: {e}")
        return None


@st.cache_data
def load_faiss_index():
    """Load the FAISS index and bottle IDs"""
    if os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(IDS_PATH, 'rb') as f:
            ids = pickle.load(f)
        return index, ids
    else:
        return None, None

# ==============================
# Improved Multi-Spirit Recommendation Engine
# ==============================


def compute_hybrid_recommendations(df, content_features, id_to_idx,
                                   faiss_index, faiss_ids, user_ids,
                                   content_weight=0.7, visual_weight=0.3,
                                   encoder=None):
    """
    Compute recommendations using both content-based and visual similarity with improved
    multi-spirit handling

    Parameters:
    -----------
    df : DataFrame
        The complete bottle dataset
    content_features : array
        Content-based feature vectors for all bottles
    id_to_idx : dict
        Mapping from bottle ID to index in content_features
    faiss_index : faiss.Index or None
        FAISS index for visual similarity search
    faiss_ids : list or None
        List of bottle IDs corresponding to FAISS index
    user_ids : list
        List of bottle IDs in the user's collection
    content_weight : float
        Weight for content-based similarity (default: 0.7)
    visual_weight : float
        Weight for visual similarity (default: 0.3)
    encoder : OneHotEncoder
        Encoder used for spirit_type, for analyzing feature importance

    Returns:
    --------
    tuple
        (general_recommendations, bottle_specific_recommendations)
    """
    # Check if user has bottles
    mask = df['id'].isin(user_ids)
    if mask.sum() == 0:
        return [], {}

    # Create a copy of the dataframe for recommendations
    rec_df = df.copy()

    # Get user's collection information
    user_df = df[mask].copy()
    user_spirit_types = user_df['spirit_type'].unique()

    # === ANALYZE USER'S SPIRIT TYPE PREFERENCES ===
    # Calculate the distribution of spirit types in user's collection
    spirit_type_counts = user_df['spirit_type'].value_counts(normalize=True)

    # === CONTENT-BASED SIMILARITY WITH SPIRIT TYPE BALANCING ===
    # Get user's bottle indices
    user_indices = [id_to_idx.get(uid) for uid in user_ids if uid in id_to_idx]

    if user_indices:
        # Create a profile vector by averaging bottle vectors
        profile_vec = content_features[user_indices].mean(axis=0)
        content_sims = cosine_similarity(
            [profile_vec], content_features).flatten()
        rec_df['similarity_content'] = content_sims

        # Boost recommendations of underrepresented spirit types
        # This helps diversify recommendations across spirit types
        if len(user_spirit_types) > 1:
            # Get the least represented spirit type
            min_spirit = spirit_type_counts.idxmin()
            # Slightly boost bottles of this type (10% boost)
            boost_mask = rec_df['spirit_type'] == min_spirit
            rec_df.loc[boost_mask, 'similarity_content'] = rec_df.loc[boost_mask,
                                                                      'similarity_content'] * 1.1
    else:
        rec_df['similarity_content'] = 0

    # === VISUAL SIMILARITY ===
    if faiss_index is not None and faiss_ids is not None:
        # Find user's bottles in the FAISS index
        visual_user_indices = []
        for uid in user_ids:
            try:
                if uid in faiss_ids:
                    visual_user_indices.append(faiss_ids.index(uid))
            except ValueError:
                continue  # Skip if not found

        if visual_user_indices:
            try:
                # Get the vectors for user's bottles
                user_vectors = np.vstack(
                    [faiss_index.reconstruct(idx) for idx in visual_user_indices])

                # Average the vectors to create a profile
                profile_vector = np.mean(user_vectors, axis=0).reshape(1, -1)

                # Search for similar bottles
                distances, indices = faiss_index.search(
                    profile_vector, len(faiss_ids))

                # Convert distances to similarities (1 - normalized distance)
                max_dist = np.max(distances) if distances.size > 0 else 1.0
                if max_dist == 0:  # Prevent division by zero
                    max_dist = 1.0

                # Create similarity mapping
                visual_sim_map = {}
                for idx, dist in zip(indices[0], distances[0]):
                    if idx < len(faiss_ids):
                        bottle_id = faiss_ids[idx]
                        similarity = 1.0 - (dist / max_dist)
                        visual_sim_map[bottle_id] = float(
                            similarity)  # Ensure it's a Python float

                # Add visual similarity scores to dataframe
                rec_df['similarity_visual'] = rec_df['id'].map(
                    visual_sim_map).fillna(0)
            except Exception as e:
                print(f"Error in visual similarity calculation: {e}")
                rec_df['similarity_visual'] = 0
        else:
            rec_df['similarity_visual'] = 0
    else:
        rec_df['similarity_visual'] = 0

    # Ensure we have at least some scores
    if rec_df['similarity_content'].max() == 0 and rec_df['similarity_visual'].max() == 0:
        # Fall back to a multi-spirit scoring approach based on user's collection
        rec_df['similarity_content'] = rec_df['spirit_type'].apply(
            lambda x: 0.8 if x in user_spirit_types else 0.2)

        # Add a price similarity component
        user_avg_price = user_df['avg_msrp'].mean()
        rec_df['price_diff'] = abs(rec_df['avg_msrp'] - user_avg_price)
        max_price_diff = rec_df['price_diff'].max(
        ) if rec_df['price_diff'].max() > 0 else 1
        rec_df['price_sim'] = 1 - (rec_df['price_diff'] / max_price_diff)

        # Combine spirit type and price similarity
        rec_df['similarity_content'] = 0.7 * \
            rec_df['similarity_content'] + 0.3 * rec_df['price_sim']

    # === COMBINE SCORES WITH WEIGHTS ===
    # Normalize weights
    total_weight = content_weight + visual_weight
    if total_weight > 0:
        norm_content_weight = content_weight / total_weight
        norm_visual_weight = visual_weight / total_weight
    else:
        norm_content_weight = 0.5
        norm_visual_weight = 0.5

    # Combine scores
    rec_df['similarity_overall'] = (
        norm_content_weight * rec_df['similarity_content'] +
        norm_visual_weight * rec_df['similarity_visual']
    )

    # === GENERATE GENERAL RECOMMENDATIONS ===
    # Get the top recommendations across all spirit types
    overall_recs = (
        rec_df[~rec_df['id'].isin(user_ids)]
        .nlargest(20, 'similarity_overall')
    )

    # Ensure diversity by selecting top recommendations from each spirit type in user's collection
    # Plus some recommendations of new spirit types for exploration
    diverse_recs = []

    # First, add top recommendation for each spirit type in user's collection
    for spirit in user_spirit_types:
        spirit_recs = overall_recs[overall_recs['spirit_type'] == spirit].head(
            1)
        if not spirit_recs.empty:
            diverse_recs.append(spirit_recs)

    # Then, add recommendations for new spirit types (exploration)
    new_spirit_recs = overall_recs[~overall_recs['spirit_type'].isin(
        user_spirit_types)].head(1)
    if not new_spirit_recs.empty:
        diverse_recs.append(new_spirit_recs)

    # Finally, add remaining top recommendations to reach 5 total
    remaining_count = 5 - len(diverse_recs)
    if remaining_count > 0:
        # Exclude bottles already selected for diverse_recs
        selected_ids = [rec['id'].values[0]
                        for rec in diverse_recs if not rec.empty]
        remaining_recs = overall_recs[~overall_recs['id'].isin(
            selected_ids)].head(remaining_count)
        if not remaining_recs.empty:
            diverse_recs.append(remaining_recs)

    # Combine all recommendations and keep top 5
    general_df = pd.concat(diverse_recs) if diverse_recs else pd.DataFrame()
    general_df = general_df.nlargest(5, 'similarity_overall')

    # Convert to the expected format
    general = (
        general_df
        [['id', 'name', 'spirit_type', 'avg_msrp', 'similarity_overall',
            'similarity_content', 'similarity_visual']]
        .rename(columns={'name': 'bottle', 'spirit_type': 'spirit_type',
                         'avg_msrp': 'msrp', 'similarity_overall': 'score',
                         'similarity_content': 'content_score',
                         'similarity_visual': 'visual_score'})
        .to_dict(orient='records')
    )

    # === GENERATE PER-BOTTLE RECOMMENDATIONS ===
    by_bottle = {}
    for bid in user_ids:
        bottle_rec_df = rec_df.copy()
        source_bottle = df[df['id'] == bid]

        if source_bottle.empty:
            continue

        source_spirit = source_bottle['spirit_type'].values[0]

        # === CONTENT-BASED SIMILARITY FOR THIS BOTTLE ===
        content_idx = id_to_idx.get(bid)
        if content_idx is not None:
            try:
                # Use cosine similarity between this bottle and all others
                bottle_vec = content_features[content_idx].reshape(1, -1)
                content_sims = cosine_similarity(
                    bottle_vec, content_features).flatten()
                bottle_rec_df['sim_content_temp'] = content_sims

                # Boost recommendations of the same spirit type
                # This ensures recommendations are similar to the specific bottle
                same_spirit_mask = bottle_rec_df['spirit_type'] == source_spirit
                bottle_rec_df.loc[same_spirit_mask, 'sim_content_temp'] = \
                    bottle_rec_df.loc[same_spirit_mask,
                                      'sim_content_temp'] * 1.2
            except Exception as e:
                print(f"Error in content similarity for bottle {bid}: {e}")
                bottle_rec_df['sim_content_temp'] = 0
        else:
            bottle_rec_df['sim_content_temp'] = 0

        # === VISUAL SIMILARITY FOR THIS BOTTLE ===
        if faiss_index is not None and faiss_ids is not None:
            try:
                # Find this bottle in the FAISS index
                if bid in faiss_ids:
                    faiss_idx = faiss_ids.index(bid)

                    # Get the bottle's vector and search for similar bottles
                    bottle_vector = faiss_index.reconstruct(
                        faiss_idx).reshape(1, -1)
                    distances, indices = faiss_index.search(
                        bottle_vector, len(faiss_ids))

                    # Convert distances to similarities
                    max_dist = np.max(distances) if distances.size > 0 else 1.0
                    if max_dist == 0:  # Prevent division by zero
                        max_dist = 1.0

                    # Create similarity mapping
                    visual_sim_map = {}
                    for idx, dist in zip(indices[0], distances[0]):
                        if idx < len(faiss_ids):
                            bottle_id = faiss_ids[idx]
                            similarity = 1.0 - (dist / max_dist)
                            visual_sim_map[bottle_id] = float(similarity)

                    # Add visual similarity scores
                    bottle_rec_df['sim_visual_temp'] = bottle_rec_df['id'].map(
                        visual_sim_map).fillna(0)
                else:
                    bottle_rec_df['sim_visual_temp'] = 0
            except Exception as e:
                print(f"Error in visual similarity for bottle {bid}: {e}")
                bottle_rec_df['sim_visual_temp'] = 0
        else:
            bottle_rec_df['sim_visual_temp'] = 0

        # Ensure we have at least some scores
        if bottle_rec_df['sim_content_temp'].max() == 0 and bottle_rec_df['sim_visual_temp'].max() == 0:
            # Fall back to a spirit-type based scoring system
            # Heavily favor the same spirit type as the source bottle
            bottle_rec_df['sim_content_temp'] = bottle_rec_df['spirit_type'].apply(
                lambda x: 0.9 if x == source_spirit else 0.3)

            # Add a price similarity component
            source_price = source_bottle['avg_msrp'].values[0]
            bottle_rec_df['price_diff'] = abs(
                bottle_rec_df['avg_msrp'] - source_price)
            max_price_diff = bottle_rec_df['price_diff'].max(
            ) if bottle_rec_df['price_diff'].max() > 0 else 1
            bottle_rec_df['price_sim'] = 1 - \
                (bottle_rec_df['price_diff'] / max_price_diff)

            # Combine spirit type and price similarity
            bottle_rec_df['sim_content_temp'] = 0.7 * \
                bottle_rec_df['sim_content_temp'] + \
                0.3 * bottle_rec_df['price_sim']

        # === COMBINE SCORES WITH WEIGHTS ===
        bottle_rec_df['sim_overall_temp'] = (
            norm_content_weight * bottle_rec_df['sim_content_temp'] +
            norm_visual_weight * bottle_rec_df['sim_visual_temp']
        )

        # === GET RECOMMENDATIONS FOR THIS BOTTLE ===
        # First get overall top recommendations
        overall_bottle_recs = (
            bottle_rec_df[~bottle_rec_df['id'].isin(user_ids)]
            .nlargest(10, 'sim_overall_temp')
        )

        # Then ensure we get some recommendations of the same spirit type
        same_spirit_recs = overall_bottle_recs[overall_bottle_recs['spirit_type'] == source_spirit].head(
            3)
        diff_spirit_recs = overall_bottle_recs[overall_bottle_recs['spirit_type'] != source_spirit].head(
            2)

        # Combine recommendations
        bottle_recs_df = pd.concat([same_spirit_recs, diff_spirit_recs])
        bottle_recs_df = bottle_recs_df.nlargest(5, 'sim_overall_temp')

        # Convert to the expected format
        bottle_recs = (
            bottle_recs_df
            [['id', 'name', 'spirit_type', 'avg_msrp', 'sim_overall_temp',
                'sim_content_temp', 'sim_visual_temp']]
            .rename(columns={'name': 'bottle', 'spirit_type': 'spirit_type',
                             'avg_msrp': 'msrp', 'sim_overall_temp': 'score',
                             'sim_content_temp': 'content_score',
                             'sim_visual_temp': 'visual_score'})
            .to_dict(orient='records')
        )

        by_bottle[bid] = bottle_recs

    return general, by_bottle

# ==============================
# User API Access
# ==============================


def fetch_user_bar(username: str):
    """Fetch the user's bar data from the Baxus API"""
    resp = requests.get(f"{API_BASE}{username}")
    resp.raise_for_status()
    return resp.json()

# ==============================
# Streamlit API Endpoint
# ==============================


def serve_api():
    """Serve recommendations as JSON via query parameters"""
    # Read query parameters
    qp = st.query_params
    username = qp.get("username") or qp.get("user") or None

    if username:
        try:
            user_bar = fetch_user_bar(username)
        except Exception as e:
            st.error(f"Error fetching user bar for '{username}': {e}")
            st.stop()

        user_ids = [item['product']['id'] for item in user_bar]

        # Get content weights from query params or use defaults
        content_weight = float(qp.get("content_weight", 0.7))
        visual_weight = float(qp.get("visual_weight", 0.3))

        # Normalize weights
        total = content_weight + visual_weight
        if total > 0:
            content_weight /= total
            visual_weight /= total

        # Compute recommendations
        general, by_bottle = compute_hybrid_recommendations(
            df, content_features, id_to_idx,
            faiss_index, faiss_ids, user_ids,
            content_weight, visual_weight,
            encoder
        )

        # Output JSON
        st.json({
            "general": general,
            "by_bottle": by_bottle,
            "meta": {
                "content_weight": content_weight,
                "visual_weight": visual_weight,
                "user_bottles_count": len(user_ids)
            }
        })
        st.stop()

# ==============================
# Main Application
# ==============================


def main():
    """Main application entry point"""
    st.title("🥃 Bob: Whisky Recommendation Agent")
    st.write("""
    Bob is your personal whisky sommelier! He analyzes your virtual bar and provides personalized recommendations 
    based on both the characteristics of your bottles and their visual appearance.
    """)

    # Configure the sidebar
    setup_sidebar()

    # Check for API mode
    serve_api()

    # Main user interface
    username = st.text_input("Enter your Baxus username:")

    if username:
        process_user_request(username)


def setup_sidebar():
    """Set up the sidebar with configuration options"""
    st.sidebar.title("Configuration")

    # Weight sliders
    st.sidebar.subheader("Recommendation Weights")
    content_weight = st.sidebar.slider(
        "Content-Based Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="How much to prioritize bottle characteristics (spirit type, price, ABV, proof)"
    )
    visual_weight = st.sidebar.slider(
        "Visual Similarity Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help="How much to prioritize bottle appearance (label design, color, shape)"
    )

    # Normalize weights
    total = content_weight + visual_weight
    if total > 0:
        content_weight_norm = content_weight / total
        visual_weight_norm = visual_weight / total
    else:
        content_weight_norm = 0.5
        visual_weight_norm = 0.5

    st.sidebar.write(
        f"Normalized weights: Content={content_weight_norm:.2f}, Visual={visual_weight_norm:.2f}")

    # Store the weights in session state
    st.session_state.content_weight = content_weight_norm
    st.session_state.visual_weight = visual_weight_norm

    # Add about section
    with st.sidebar.expander("About Bob"):
        st.write("""
        Bob is a hybrid recommendation system for whisky enthusiasts. It combines:
        
        1. **Content-based filtering** - Analyzing spirit type, price, ABV, and proof
        2. **Visual similarity** - Comparing bottle appearance using deep learning
        
        Created for the BAXUS ecosystem to help users discover new bottles for their collection.
        """)


def process_user_request(username):
    """Process a user's request for recommendations"""
    try:
        with st.spinner("Fetching your bar data..."):
            user_bar = fetch_user_bar(username)
    except Exception as e:
        st.error(f"Error fetching data for '{username}': {e}")
        st.stop()

    if not user_bar:
        st.warning("No bottles found in your bar.")
        return

    # Display user's bar
    display_user_bar(user_bar)

    # Compute and display recommendations
    with st.spinner("Analyzing your collection and generating recommendations..."):
        user_ids = [item['product']['id'] for item in user_bar]
        general, by_bottle = compute_hybrid_recommendations(
            df, content_features, id_to_idx,
            faiss_index, faiss_ids, user_ids,
            st.session_state.content_weight,
            st.session_state.visual_weight,
            encoder
        )

    # Display recommendations
    display_general_recommendations(general)
    display_bottle_recommendations(by_bottle, user_bar)


def display_user_bar(user_bar):
    """Display the user's bar overview"""
    # Extract bottle data
    bar_records = []
    for item in user_bar:
        prod = item.get('product', {})
        bar_records.append({
            'id': prod.get('id'),
            'Bottle': prod.get('name', 'Unknown'),
            'Spirit': prod.get('spirit', 'Unknown'),
            'Price (MSRP)': prod.get('average_msrp', 0),
            'Proof': prod.get('proof', None),
            'Image': prod.get('image_url')
        })

    bar_df = pd.DataFrame(bar_records)

    # Display bar overview
    st.subheader("Your Bar Overview")

    # Group by spirit type to show collection composition
    if 'Spirit' in bar_df.columns and not bar_df.empty:
        spirit_counts = bar_df['Spirit'].value_counts()
        st.write(
            f"Your collection has {len(bar_df)} bottles across {len(spirit_counts)} spirit types.")

        # Show spirit type distribution in columns
        cols = st.columns(min(len(spirit_counts), 4))
        for i, (spirit, count) in enumerate(spirit_counts.items()):
            with cols[i % 4]:
                st.metric(spirit, count)

    # Show bottles in table
    st.dataframe(
        bar_df[['Bottle', 'Spirit', 'Price (MSRP)', 'Proof']],
        use_container_width=True
    )


def display_general_recommendations(general):
    """
    Display general recommendations

    Parameters:
    -----------
    general : list
        List of general recommendations
    """
    st.subheader("General Recommendations")

    if not general:
        st.info("No general recommendations available.")
        return

    # Convert to DataFrame for detailed view
    gen_df = pd.DataFrame(general).rename(columns={
        'bottle': 'Bottle', 'spirit_type': 'Spirit Type',
        'msrp': 'Price (MSRP)', 'score': 'Match Score',
        'content_score': 'Content Score', 'visual_score': 'Visual Score'
    })

    # Create columns for card layout
    cols = st.columns(min(len(general), 5))
    for i, (col, rec) in enumerate(zip(cols, general)):
        with col:
            # Create styled recommendation card
            st.markdown(f"### {rec['bottle']}")
            st.markdown(f"**{rec['spirit_type']}** - ${rec['msrp']:.2f}")

            # Use color coding for match score
            score = rec['score']
            score_color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
            st.markdown(
                f"**Match Score:** <span style='color:{score_color}'>{score:.2f}</span>", unsafe_allow_html=True)

            # Score breakdown with bar visualization
            cont_score = rec.get('content_score', 0)
            vis_score = rec.get('visual_score', 0)

            st.markdown("**Score Breakdown:**")
            st.markdown(f"Content: {cont_score:.2f} | Visual: {vis_score:.2f}")

    # Show detailed table
    with st.expander("View Detailed Recommendation Data"):
        st.dataframe(gen_df, use_container_width=True)


def display_bottle_recommendations(by_bottle, user_bar):
    """
    Display bottle-specific recommendations

    Parameters:
    -----------
    by_bottle : dict
        Dictionary of bottle-specific recommendations
    user_bar : list
        User's bar data from API
    """
    st.subheader("Recommendations Based on Bottle")

    if not by_bottle:
        st.info("No bottle-specific recommendations available.")
        return

    # Create a mapping of bottle IDs to names
    bottle_names = {}
    bottle_data = {}
    for item in user_bar:
        prod = item.get('product', {})
        bottle_id = prod.get('id')
        bottle_name = prod.get('name', 'Unknown Bottle')
        if bottle_id:
            bottle_names[bottle_id] = bottle_name
            bottle_data[bottle_id] = prod

    # Create tabs for each bottle
    tabs = st.tabs(
        [bottle_names.get(bid, f"Bottle {bid}") for bid in by_bottle.keys()])

    for tab, (bid, recs) in zip(tabs, by_bottle.items()):
        with tab:
            if not recs:
                st.info(
                    f"No specific recommendations for {bottle_names.get(bid, 'this bottle')}.")
                continue

            # Show source bottle information
            if bid in bottle_data:
                source = bottle_data[bid]
                st.markdown(f"## Based on: {source.get('name')}")
                st.markdown(
                    f"**{source.get('spirit', 'Unknown')}** - ${source.get('average_msrp', 0):.2f}")

            # Convert to DataFrame
            rec_df = pd.DataFrame(recs).rename(columns={
                'bottle': 'Bottle', 'spirit_type': 'Spirit Type',
                'msrp': 'Price (MSRP)', 'score': 'Match Score',
                'content_score': 'Content Score', 'visual_score': 'Visual Score'
            })

            if not rec_df.empty:
                # Create columns for top recommendations
                num_to_show = min(3, len(recs))
                rec_cols = st.columns(num_to_show)

                for i in range(num_to_show):
                    if i < len(recs):
                        with rec_cols[i]:
                            rec = recs[i]

                            st.markdown(f"### {rec['bottle']}")
                            st.markdown(
                                f"**{rec['spirit_type']}** - ${rec['msrp']:.2f}")

                            # Use color coding for match score
                            score = rec['score']
                            score_color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
                            st.markdown(
                                f"**Match:** <span style='color:{score_color}'>{score:.2f}</span>", unsafe_allow_html=True)

                # Show all recommendations in a table
                st.markdown("### All Recommendations")
                st.dataframe(
                    rec_df[['Bottle', 'Spirit Type',
                            'Price (MSRP)', 'Match Score', 'Content Score', 'Visual Score']],
                    use_container_width=True
                )

# ==============================
# App Initialization
# ==============================


if __name__ == '__main__':
    # Load dataset
    df = load_dataset()
    content_features, id_to_idx, encoder = prepare_content_features(df)

    # Load FAISS index
    with st.spinner("Loading visual similarity model..."):
        faiss_index, faiss_ids = load_faiss_index()

    # Initialize default weights if not set
    if 'content_weight' not in st.session_state:
        st.session_state.content_weight = 0.7
    if 'visual_weight' not in st.session_state:
        st.session_state.visual_weight = 0.3

    # Run the main app
    main()
