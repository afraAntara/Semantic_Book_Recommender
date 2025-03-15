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

### Test 1 to check if Embeddings Exist ###
def test_embeddings_exist():
    """
    Test to check if the book embeddings are correctly generated and loaded.
    """
    assert isinstance(book_embeddings, torch.Tensor), "Embeddings should be a PyTorch tensor."
    assert book_embeddings.shape[0] > 0, "Embeddings tensor should not be empty."


### Test 2 to Inference & Latency Times ###
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


### Test 3 to check if Streamlit Returns Correct Recommendations ###
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
