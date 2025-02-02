# Semantic Book Recommender

## Overview
Semantic Book Recommender is a web app that helps users find books based on meaning (semantics) rather than just keywords. It uses NLP to understand what users are looking for and suggests books that match their interests.

## Features
- **Smart Book Search**: Finds books based on meaning, not just keywords.
- **Category Filtering**: Users can filter books by category.
- **Similarity Scoring**: Shows how closely recommended books match the search query.
- **Fast and Efficient**: Uses a pre-trained model to provide quick results.

## How It Works
1. The app loads book data from a CSV file.
2. It combines book details (title, author, description, category, and publisher) into a single text entry.
3. It uses the `SentenceTransformer` model to convert book details into numerical embeddings.
4. When a user enters a query, the app computes its embedding and compares it to book embeddings using cosine similarity.
5. The top three most relevant books are displayed as recommendations.

## How to Run the App
1. Install the required Python libraries:
   ```bash
   pip install streamlit pandas numpy torch sentence-transformers
   ```
2. Place your book dataset (`book_data.csv`) in the project folder.
3. Run the app using Streamlit:
   ```bash
   streamlit run app.py
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




