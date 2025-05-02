# Bob: Hybrid Whisky Recommendation Agent

Bob is a Streamlit-based AI/ML recommendation system that analyzes a user's virtual spirits collection (via the Baxus API) and provides personalized bottle recommendations. Using both content-based filtering and visual similarity analysis, Bob delivers diverse suggestions tailored to your unique collection.

## Features

- **Hybrid Recommendation Engine**: Combines traditional content-based filtering with cutting-edge visual similarity using deep learning.
- **Multi-Spirit Intelligence**: Analyzes your entire collection across all spirit types, not just whisky.
- **Personalized Recommendations**: Two recommendation types:
  - **General Recommendations**: Top 5 suggestions based on your entire collection profile, with intelligent diversity across spirit types.
  - **Per-Bottle Recommendations**: For each bottle in your collection, get 5 similar bottles with a balance of same-spirit and complementary recommendations.
- **Visual Similarity Analysis**: Uses deep learning (MobileNetV2) and FAISS to compare bottle appearance and design elements.
- **Content-Based Filtering**: Analyzes spirit type, price point, ABV, and proof to find matches based on characteristics.
- **Smart Diversity**: Ensures recommendations include bottles from each spirit category in your collection plus discovery suggestions.
- **Interactive UI**: Clean, visual interface with detailed scoring breakdowns.
- **Live API Integration**: Fetches bar data in real time from `https://services.baxus.co/api/bar/user/{username}`.
- **REST API Endpoint**: Supports direct recommendation retrieval via URL parameters.

## Requirements

- Python 3.8+
- Streamlit
- pandas
- scikit-learn
- requests
- tensorflow
- faiss-cpu
- pillow
- matplotlib (for index building)
- tqdm (for progress bars)

Install via:
```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the repo**

```bash
git clone https://github.com/Code-Parth/ai-agent-bob-baxus.git
cd ai-agent-bob-baxus
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Place the dataset**
   
   Ensure `501-Bottle-Dataset.csv` is in the project root. The dataset should include at minimum:
   - `id`: Unique identifier for each bottle
   - `name`: Name of the bottle
   - `spirit_type`: Type of spirit (e.g., Bourbon, Scotch, Gin)
   - `avg_msrp`: Average retail price
   - `abv`: Alcohol by volume percentage
   - `proof`: Proof value
   - `image_url`: URL to bottle image (for visual similarity)

4. **Build the visual similarity index (optional but recommended)**

```bash
python build_index.py
```

This process:
- Downloads bottle images from URLs in the dataset
- Extracts visual features using MobileNetV2
- Builds a FAISS index for fast similarity search
- Saves the index and bottle IDs for use by the main application

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

1. Enter your Baxus username in the input field.
2. View your bar overview with spirit type distribution.
3. Adjust content vs. visual similarity weights in the sidebar if desired.
4. Explore your personalized recommendations:
   - **General Recommendations**: Diverse suggestions based on your entire collection
   - **Per-Bottle Recommendations**: Specific recommendations for each bottle in your collection

## How It Works

### Hybrid Recommendation System

Bob uses a sophisticated hybrid approach combining two recommendation strategies:

1. **Content-Based Filtering**:
   - Analyzes categorical features (spirit type) using one-hot encoding
   - Processes numerical features (price, ABV, proof) with normalization
   - Creates feature vectors for each bottle
   - Uses cosine similarity to find similar bottles

2. **Visual Similarity**:
   - Extracts deep features from bottle images using MobileNetV2
   - Builds a FAISS index for efficient similarity search
   - Compares bottle appearance, label design, and packaging

3. **Intelligent Combination**:
   - Weights both approaches according to user preference
   - Ensures diversity across spirit categories
   - Boosts underrepresented spirit types in your collection
   - Balances recommendations between familiar and discovery options

### Multi-Spirit Intelligence

The system analyzes your collection across all spirit types and ensures recommendations include:
- Representatives from each spirit type in your collection
- Suggestions for new spirit types to try (exploration)
- Proper balance between same-spirit and cross-category recommendations

For bottle-specific recommendations, the system ensures you get mostly recommendations of the same spirit type (e.g., similar bourbons for a bourbon) but also includes some cross-category recommendations for discovery.

## API Mode: Integrate with the Website

Bob supports a queryable API interface for embedding recommendation data:

### Access JSON Recommendations:
```
http://localhost:8501/?username=<your_username>&content_weight=0.7&visual_weight=0.3
```

### Parameters:
- `username` (required): Baxus username
- `content_weight` (optional): Weight for content-based similarity (default: 0.7)
- `visual_weight` (optional): Weight for visual similarity (default: 0.3)

### Example JSON Response:
```json
{
  "general": [
    {"bottle": "Ardbeg 10", "spirit_type": "Scotch", "msrp": 54.0, "score": 0.92, "content_score": 0.94, "visual_score": 0.87 },
    ...
  ],
  "by_bottle": {
    "13266": [
      {"bottle": "E.H. Taylor Small Batch", "spirit_type": "Bourbon", "msrp": 44.99, "score": 0.89, "content_score": 0.91, "visual_score": 0.82 },
      ...
    ]
  },
  "meta": {
    "content_weight": 0.7,
    "visual_weight": 0.3,
    "user_bottles_count": 5
  }
}
```

## Customization

- **Recommendation Weights**: Adjust the balance between content-based and visual similarity using the sidebar sliders.
- **Add New Features**: Extend `prepare_content_features()` to incorporate additional bottle characteristics.
- **Visual Index Quality**: Add more bottle images to improve visual similarity.
- **Recommendation Diversity**: Modify the diversity parameters in the recommendation engine.

## License

This project is provided under the MIT License. Feel free to fork and adapt for your own spirits recommendation needs!