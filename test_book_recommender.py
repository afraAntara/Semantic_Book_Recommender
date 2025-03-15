import pytest
import numpy as np
import pandas as pd
import torch
import time
import sys
from book_recommender import (
    load_local_dataset,
    compute_embeddings,
    compute_query_embedding,
    get_top_recommendations,
    get_book_embeddings,
    model,
)


# Load dataset and embeddings
df = load_local_dataset()
book_embeddings = get_book_embeddings()

# Prevents starting streamlit UI
if "pytest" in sys.modules:
    import streamlit as st
    from unittest.mock import MagicMock
    st.write = MagicMock()  # Mock Streamlit functions
    st.subheader = MagicMock()
    st.text_input = MagicMock(return_value="Test Query")

### Test 1: Check if Dataset Loads Correctly ###
def test_load_local_dataset():
    """Test that the dataset is loaded correctly and contains required columns."""
    required_columns = ["Title", "Authors", "Description", "Category", "Publisher", "combined_text"]
    
    assert isinstance(df, pd.DataFrame), "Dataset should be a Pandas DataFrame."
    assert not df.empty, "Dataset should not be empty."
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

### Test 2 to check if Embeddings Exist ###
def test_embeddings_exist():
    """
    Test to check if the book embeddings are correctly generated and loaded.
    """
    assert isinstance(book_embeddings, torch.Tensor), "Embeddings should be a PyTorch tensor."
    assert book_embeddings.shape[0] > 0, "Embeddings tensor should not be empty."



### Test 3: Check Embedding Computation ###
def test_compute_embeddings():
    """Test that embeddings are computed and saved correctly."""
    test_embeddings = compute_embeddings(df)
    
    assert isinstance(test_embeddings, torch.Tensor), "Embeddings should be a PyTorch tensor."
    assert test_embeddings.shape[0] == len(df), "Embeddings count should match dataset size."

### Test 4: Check Query Embedding ###
def test_compute_query_embedding():
    """Test that query embeddings are correctly generated."""
    user_query = "A fantasy adventure book"
    query_embedding = compute_query_embedding(user_query)
    
    assert query_embedding is not None, "Query embedding should not be None."
    assert isinstance(query_embedding, torch.Tensor), "Query embedding should be a PyTorch tensor."

### Test 5 to Inference & Latency Times ###
def test_inference_latency_time():
    """
    Test to check if inference (query embedding) and latency (recommendations) are computed within reasonable time.
    """
    user_query = "A mystery thriller book"
    
    # Measure query embedding time
    start_time = time.time()
    query_embedding = compute_query_embedding(user_query)
    end_time = time.time()
    
    query_latency = end_time - start_time
    assert query_embedding is not None, "Query embedding should not be None."
    assert query_latency < 1, "Query inference should be under 1 seconds."

    # Filter data for recommendations
    filtered_df = df.copy()
    filtered_df = filtered_df.reset_index(drop=True)
    embeddings_filtered = book_embeddings[filtered_df.index]

    # Measure recommendation latency
    start_time = time.time()
    top_books = get_top_recommendations(query_embedding, filtered_df, embeddings_filtered)
    end_time = time.time()

    recommendation_latency = end_time - start_time
    assert recommendation_latency < 1, "Recommendation latency should be under 1 seconds."


### Test 6 to check if Streamlit Returns Correct Recommendations ###
def test_recommendation_output():
    """
    Test to check if the recommendation function returns a DataFrame with expected structure and values.
    """
    user_query = "Science fiction novel about space travel"
    query_embedding = compute_query_embedding(user_query)

    filtered_df = df.copy()
    filtered_df = filtered_df.reset_index(drop=True)
    embeddings_filtered = book_embeddings[filtered_df.index]

    top_books = get_top_recommendations(query_embedding, filtered_df, embeddings_filtered)

    # Assertions
    assert isinstance(top_books, pd.DataFrame), "Recommendations should be a Pandas DataFrame."
    assert not top_books.empty, "Recommendation output should not be empty."
    assert "Title" in top_books.columns, "Output DataFrame must have 'Title' column."
    assert "similarity" in top_books.columns, "Output DataFrame must have 'similarity' column."
    assert top_books["similarity"].iloc[0] >= top_books["similarity"].iloc[-1], "Books should be ranked by similarity score."









