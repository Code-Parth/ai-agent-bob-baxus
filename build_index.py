import os
import pickle
import faiss
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# === CONFIGURATION ===
CSV_PATH = '501-Bottle-Dataset.csv'
URL_COL = 'image_url'
ID_COL = 'id'
IMAGE_DIR = 'bottle_images'
INDEX_PATH = 'bottle_index.faiss'
IDS_PATH = 'bottle_ids.pkl'
METADATA_PATH = 'bottle_metadata.csv'


def main():
    """Main execution function"""
    print("🥃 Bob: Building Whisky Visual Similarity Index 🥃")

    # Create image directory if it doesn't exist
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Step 1: Load and preprocess metadata
    print("\n=== 1) Loading Bottle Dataset ===")
    df = load_and_preprocess_metadata()

    # Step 2: Download bottle images
    print("\n=== 2) Downloading Bottle Images ===")
    download_bottle_images(df)

    # Step 3: Initialize feature extractor model
    print("\n=== 3) Loading Feature Extractor ===")
    model = load_feature_extractor()

    # Step 4: Build FAISS index
    print("\n=== 4) Building FAISS Index ===")
    build_faiss_index(df, model)

    print("\n✅ All done! The index is ready to use with bob_app.py")


def load_and_preprocess_metadata():
    """Load and preprocess the bottle dataset"""
    print("Loading dataset from", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Verify required columns exist
    required_cols = [ID_COL, URL_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")

    # Clean and preprocess
    print("Preprocessing dataset...")
    df['avg_msrp'] = pd.to_numeric(df['avg_msrp'], errors='coerce').fillna(0)
    df['abv'] = pd.to_numeric(df['abv'], errors='coerce').fillna(0)
    df['proof'] = pd.to_numeric(
        df['proof'], errors='coerce').fillna(df['abv'] * 2)

    # Save clean metadata
    df.to_csv(METADATA_PATH, index=False)
    print(f"✓ Saved clean metadata to {METADATA_PATH}")
    print(f"✓ Dataset contains {len(df)} bottles")

    return df


def download_bottle_images(df):
    """Download bottle images from URLs in the dataset"""
    print(f"Downloading images to {IMAGE_DIR}...")

    success_count = 0
    skip_count = 0
    error_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        url = row.get(URL_COL)
        bid = row.get(ID_COL)

        # Skip if URL or ID is missing
        if pd.isna(url) or pd.isna(bid):
            continue

        # Clean URL
        url = url.strip()
        if not url.lower().startswith(('http://', 'https://')):
            continue

        # Define file path
        ext = os.path.splitext(url)[1].split('?')[0] or '.jpg'
        fn = f"{int(bid)}{ext}"
        out = os.path.join(IMAGE_DIR, fn)

        # Skip if image already exists
        if os.path.exists(out):
            skip_count += 1
            continue

        # Download image
        try:
            r = requests.get(url, stream=True, timeout=10)
            r.raise_for_status()
            with open(out, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            success_count += 1
        except Exception as e:
            print(f"⚠️ Failed to download image for {bid}: {e}")
            error_count += 1

    print(f"✓ Downloaded {success_count} new images")
    print(f"✓ Skipped {skip_count} existing images")
    print(f"✓ Failed to download {error_count} images")

    # Count total valid images
    total_images = len([f for f in os.listdir(IMAGE_DIR)
                       if os.path.isfile(os.path.join(IMAGE_DIR, f))])
    print(f"✓ Total images in directory: {total_images}")


def load_feature_extractor():
    """Load the MobileNetV2 model for feature extraction"""
    print("Loading MobileNetV2...")
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    print("✓ Model loaded")
    return model


def extract_features(img_path, model):
    """Extract features from an image using MobileNetV2"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))
    feats = model.predict(x, verbose=0)  # Suppress verbose output
    return feats[0].astype('float32')


def build_faiss_index(df, model):
    """Build a FAISS index from bottle images"""
    features = []
    ids = []
    files_processed = 0
    failed_files = []

    print(f"Processing images from {IMAGE_DIR}...")

    for fn in tqdm(os.listdir(IMAGE_DIR), desc="Extracting features"):
        if not os.path.isfile(os.path.join(IMAGE_DIR, fn)):
            continue

        try:
            # Extract bottle ID from filename
            bid = int(os.path.splitext(fn)[0])

            # Skip if not in our dataset
            if bid not in df[ID_COL].values:
                continue

            # Extract features
            img_path = os.path.join(IMAGE_DIR, fn)
            feats = extract_features(img_path, model)

            # Add to our lists
            features.append(feats)
            ids.append(bid)
            files_processed += 1

        except Exception as e:
            print(f"⚠️ Error processing {fn}: {e}")
            failed_files.append(fn)

    if not features:
        raise ValueError("No valid images found or processed!")

    # Build the index
    print(f"Building FAISS index with {len(features)} images...")
    features_array = np.stack(features)
    d = features_array.shape[1]  # dimensionality

    # Create and add vectors to index
    index = faiss.IndexFlatL2(d)
    index.add(features_array)

    # Save index & ids
    faiss.write_index(index, INDEX_PATH)
    with open(IDS_PATH, 'wb') as f:
        pickle.dump(ids, f)

    print(f"✓ Processed {files_processed} images")
    print(f"✓ Failed to process {len(failed_files)} images")
    print(f"✓ Indexed {index.ntotal} images with {d} dimensions")
    print(f"✓ Saved index to {INDEX_PATH}")
    print(f"✓ Saved bottle IDs to {IDS_PATH}")

    # Visualize a sample of the images
    visualize_sample_bottles(ids, df)


def visualize_sample_bottles(ids, df):
    """Visualize a sample of bottles in the index"""
    if not ids:
        return

    print("Generating sample visualization...")

    # Sample up to 5 bottle IDs
    sample_ids = ids[:5] if len(ids) > 5 else ids

    # Get bottle names
    sample_data = []
    for bid in sample_ids:
        bottle_info = df[df[ID_COL] == bid]
        if not bottle_info.empty:
            name = bottle_info['name'].values[0]
            spirit = bottle_info['spirit_type'].values[0] if 'spirit_type' in bottle_info.columns else 'Unknown'
            sample_data.append((bid, name, spirit))

    # Print sample information
    print("\nSample of indexed bottles:")
    for bid, name, spirit in sample_data:
        print(f"- {name} ({spirit}, ID: {bid})")


if __name__ == "__main__":
    main()
