import streamlit as st
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import os

# Check for CUDA availability
device = 'cpu'  # Force CPU since we want an optimized CPU version

# Load pre-trained Sentence Transformer model on CPU
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load book dataset from a local CSV file
def load_local_dataset():
    file_path = "book_data.csv"
    df = pd.read_csv(file_path)
    required_columns = ["Title", "Authors", "Description", "Category", "Publisher"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset must contain '{col}' column.")
    
    # Handle NaN values by replacing them with an empty string
    df["Description"] = df["Description"].fillna("")
    df["Category"] = df["Category"].fillna("")
    return df


df = load_local_dataset()
# Combine all relevant text features into a single column
df["combined_text"] = df["Title"] + " " + df["Authors"] + " " + df["Description"] + " " + df["Category"] + " " + df["Publisher"]

# Cache embeddings
embeddings_file = "book_embeddings.npy"

if torch.cuda.is_available():
    device = 'cuda'

if not os.path.exists(embeddings_file):
    book_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True).half()
    np.save(embeddings_file, book_embeddings.cpu().numpy())  # Save to disk

@st.cache_data
def get_book_embeddings():
    return torch.tensor(np.load(embeddings_file))

book_embeddings = get_book_embeddings()

# Streamlit UI setup
st.title("Semantic Search-Based Book Recommendation")
st.write("Enter a brief description of the type of book you're looking for:")

# Optional category filter
selected_category = st.selectbox("Filter by category:", ["All"] + df["Category"].unique().tolist())

if selected_category != "All":
    df_filtered = df[df["Category"] == selected_category]
    book_embeddings_filtered = book_embeddings[df_filtered.index]
else:
    df_filtered = df
    book_embeddings_filtered = book_embeddings

# User input
user_query = st.text_input("Search for a book:")

if user_query:
    # Compute embedding for user query
    query_embedding = model.encode([user_query], convert_to_tensor=True).half()

    # Compute similarity using PyTorch (Much Faster)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, book_embeddings_filtered)

    # Rank books by similarity score
    df_filtered["similarity"] = similarities.cpu().numpy()
    top_books = df_filtered.sort_values(by="similarity", ascending=False).head(3)

    # Display recommendations
    st.subheader("Recommended Books:")
    for _, row in top_books.iterrows():
        st.write(f"**{row['Title']}** by {row['Authors']}")
        st.write(f"Category: {row['Category']}, Publisher: {row['Publisher']}")
        st.write(f"Description: {row['Description']}")
        st.write(f"Similarity Score: {row['similarity']:.2f}")
        st.markdown("---")

