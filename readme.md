# Bob: Whisky Recommendation Agent

Bob is a Streamlit-based AI/ML prototype that analyzes a user’s virtual whisky bar (via the Baxus API) and recommends new bottles both for your overall collection and on a per-bottle basis.

## Features

- **General Recommendations**: Top 5 suggestions based on your entire bar profile.
- **Per-Bottle Suggestions**: Accordion-style (expander) panels showing the 5 most similar bottles for each item in your bar.
- **Content-Based Filtering**: Combines categorical (spirit type) and numerical (price, ABV, proof) features into vectors and ranks by cosine similarity.
- **Live API Integration**: Fetches bar data in real time from `https://services.baxus.co/api/bar/user/{username}`.
- **REST API Endpoint**: Supports direct recommendation retrieval via URL query (great for embedding into other tools or dashboards).

## Requirements

- Python 3.8+
- Streamlit
- pandas
- scikit-learn
- requests

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
   Ensure `501-Bottle-Dataset.csv` is in the project root (it is included).

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

1. Enter your Baxus username in the input field.
2. View your bar overview table.
3. Expand the **General Recommendations – Top 5** panel for overall suggestions.
4. Scroll down to see expanders for each bottle in your bar, showing its top 5 similar bottles.

## API Mode: Integrate with the Website

Bob supports a queryable API interface so you can embed recommendation data into the Baxus web frontend or a companion mobile app.

### Access JSON Recommendations:
```
http://localhost:8501/?username=<your_username>
```

### Example JSON Response:
```json
{
  "general": [
    {"bottle": "Ardbeg 10", "spirit_type": "Scotch", "msrp": 54.0, "score": 0.92 },
    ...
  ],
  "by_bottle": {
    "13266": [
      {"bottle": "E.H. Taylor Small Batch", "spirit_type": "Bourbon", "msrp": 44.99, "score": 0.89 },
      ...
    ]
  }
}
```

### Benefits for Integration:
- Populate the "Wishlist Suggestions" tab in the user’s dashboard
- Show per-bottle alternatives in product detail pages
- Use recommendations as onboarding nudges when a user adds their first bottles
- Update user experience dynamically without needing a UI refresh

## How It Works

1. **Data Loading**: Reads the 501-bottle CSV and normalizes `avg_msrp`, `abv`, and `proof`.
2. **Feature Preparation**: One-hot encodes `spirit_type` and scales numerical columns; merges into a feature matrix.
3. **Profile Vector**: Averages feature vectors of the user's bottles to form an overall taste profile.
4. **Similarity Computation**: Uses cosine similarity to score all candidate bottles against the profile (and each individual bottle).
5. **Recommendation Output**: Ranks and displays the top 5 matches in interactive tables and expanders.

## Customization

- **Add New Features**: Integrate flavor-note embeddings or age-statement buckets by extending `prepare_features()`.
- **Adjust Top-N**: Change the hardcoded `head(5)` calls to a different number.
- **Styling**: Enhance the Streamlit UI with images, custom themes, or more interactive filters.

## License

This project is provided under the MIT License. Feel free to fork and adapt for your own whisky recommendation needs!
