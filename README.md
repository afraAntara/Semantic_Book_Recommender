# Semantic Book Recommender

**Click on this link to use:** https://semanticbookrecommender-bj9hk62j29ck79hjhp6zrk.streamlit.app/

## Overview
Semantic Book Recommender is an AI-powered web application that helps users discover books based on **semantic meaning** rather than just keyword matching. It leverages **Natural Language Processing (NLP)** and **pre-trained embeddings** to understand user queries and recommend books with high relevance.

## Features
- **Smart Book Search**: Finds books based on meaning, not just keywords.
- **Category Filtering**: Users can filter books by category.
- **Similarity Scoring**: Shows how closely recommended books match the search query.
- **Fast and Efficient**: Uses a pre-trained model to provide quick results.
- **Smart Book Search** – Finds books based on **meaning**, not just keywords.
-  **Optimized for Speed** – Computes embeddings and caches them for **fast** search results.
- **A/B Testing for Recommendation Models** – Tests **Cosine Similarity vs. Dot Product Similarity** for better performance.
- **User Interaction Tracking** – Logs **views, purchases, and engagement time** in a CSV file for analysis.


## How It Works
1. **Load Book Dataset** – The app reads **book_data.csv**, containing book titles, authors, descriptions, categories, and publishers.
2. **Preprocess Text Data** – Combines relevant book details into a **single text entry** for each book.
3. **Generate Embeddings** – Uses the `SentenceTransformer` model to **convert text into numerical vectors (embeddings)**.
4. **Process User Queries**:
   - Converts the user's input into an **embedding**.
   - Computes similarity scores between the query and **all book embeddings**.
   - **Returns the top 3 most relevant books**.
5. **A/B Testing**:
   - Users are **randomly assigned** to **Variant A (Cosine Similarity)** or **Variant B (Dot Product Similarity)**.
   - Their interactions are **tracked** to evaluate which method performs better.
6. **Track User Engagement** – Logs **click-through rate (CTR), conversion rate, engagement time, and bounce rate**.

## How to Run the App
1. Install the required Python libraries:
   ```bash
   pip install streamlit pandas numpy torch sentence-transformers
   ```
2. Place your book dataset (`book_data.csv`) in the project folder.
3. Run the app using Streamlit:
   ```bash
   streamlit run book_recommender.py   
   ```
## Exploratory Data Analysis
- The dataset used is from Kaggle: [Books Dataset](https://www.kaggle.com/datasets/elvinrustam/books-dataset/data).
- The dataset originally contained 103,082 rows and 7 features.
- The features 'Publish date' and 'Price' were removed.
- To reduce computation time, only the first 5,000 books were selected for the project.
  
## Usage
- Enter a description of the book you want to find.
- (Optional) Filter results by category.
- View the top recommended books with details like author, category, publisher, and similarity score.

## Requirements
- Python 3.7+
- Streamlit
- NumPy
- Pandas
- Torch
- Sentence Transformers

## Notes
- The app is optimized for CPU usage, but it can use a GPU if available.
- If running for the first time, book embeddings will be generated and saved for faster searches in the future.

## Future Improvements
- Add more filters (e.g., author, publisher).
- Improve UI design.
- Support larger datasets.
- Allow users to save/bookmark favorite recommendations.

## License
This project is open-source. Feel free to modify and improve it!


