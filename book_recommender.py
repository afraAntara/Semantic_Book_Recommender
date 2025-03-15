import streamlit as st
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import os
import logging
import time
import random 
import csv  

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DATA_FILE = "book_data.csv"
EMBEDDINGS_FILE = "book_embeddings.npy"  # cache embeddings to prevent re-computing embeddings
USER_CLICKS_FILE = "user_interactions.csv"  # file to track clicks on view & buy
MODEL_NAME = 'all-MiniLM-L6-v2'
DEVICE = 'cpu'  

# Load pre-trained Sentence Transformer model
try:
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
except Exception as e:
    logging.error(f"Error loading SentenceTransformer model: {e}")
    raise

def load_local_dataset(filepath=DATA_FILE):
    """Loads and validates the book dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        required_columns = ["Title", "Authors", "Description", "Category", "Publisher"]
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Dataset must contain '{col}' column.")

        df.fillna("", inplace=True)
        df["combined_text"] = df["Title"] + " " + df["Authors"] + " " + df["Description"] + " " + df["Category"] + " " + df["Publisher"]
        return df

    except FileNotFoundError:
        logging.error("Dataset file not found. Please check the path.")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

df = load_local_dataset()

def compute_embeddings(df, save_path=EMBEDDINGS_FILE):
    """Computes and saves embeddings if not already stored."""
    if not os.path.exists(save_path):
        logging.info("Computing book embeddings...")
        start_time = time.time()

        try:
            book_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True).half()
            np.save(save_path, book_embeddings.cpu().numpy())

            end_time = time.time()
            logging.info(f"Book embeddings computed in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            logging.error(f"Error computing embeddings: {e}")
            raise
    
    return torch.tensor(np.load(save_path))

@st.cache_data(show_spinner=False)
def get_book_embeddings():
    """Loads book embeddings from file."""
    try:
        return torch.tensor(np.load(EMBEDDINGS_FILE))
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        raise

book_embeddings = compute_embeddings(df)

def compute_query_embedding(query):
    """Computes the embedding for a user query."""
    try:
        return model.encode([query], convert_to_tensor=True).half()
    except Exception as e:
        logging.error(f"Error encoding query: {e}")
        return None

def get_top_recommendations(query_embedding, df_filtered, embeddings_filtered, top_n=3, method="cosine"):
    """
    Retrieves top N book recommendations based on similarity.

    Args:
        query_embedding (torch.Tensor): User query embedding.
        df_filtered (pd.DataFrame): Filtered dataframe.
        embeddings_filtered (torch.Tensor): Filtered book embeddings.
        top_n (int): Number of recommendations.
        method (str): Similarity method ("cosine" or "dot_product").

    Returns:
        pd.DataFrame: Top recommended books.
    """
    try:
        start_time = time.time()

        if method == "cosine":
            similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings_filtered)
        elif method == "dot_product":
            similarities = torch.matmul(query_embedding, embeddings_filtered.T).squeeze()
        else:
            raise ValueError("Invalid method. Choose 'cosine' or 'dot_product'.")

        df_filtered["similarity"] = similarities.cpu().numpy()
        top_books = df_filtered.sort_values(by="similarity", ascending=False).head(top_n)

        end_time = time.time()
        logging.info(f"Query processed in {end_time - start_time:.2f} seconds using {method} similarity.")

        return top_books
    except Exception as e:
        logging.error(f"Error computing similarity: {e}")
        return pd.DataFrame()

def log_user_action(user_query, book_title, action, variant, start_time):
    """
    Logs user actions (View or Buy) into a CSV file for analytics.

    Args:
        user_query (str): The user's search query.
        book_title (str): The title of the book clicked.
        action (str): "View" or "Buy".
        variant (str): A/B Test variant.
        start_time (float): Timestamp when user input query.
    """
    engagement_time = time.time() - start_time  # Time since query input
    with open(USER_CLICKS_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), user_query, book_title, action, variant, round(engagement_time, 2)])

# Streamlit UI
st.title("Semantic Search-Based Book Recommendation (A/B Test)")
st.write("Search books below")

# Assign A/B test per session
if "ab_variant" not in st.session_state:
    st.session_state["ab_variant"] = random.choice(["A", "B"])  # Randomly assigns A or B once per session

ab_variant = st.session_state["ab_variant"]
st.sidebar.write(f"**A/B Test Variant: {ab_variant}**")

# Flagging new session in CSV
if "session_started" not in st.session_state:
    with open(USER_CLICKS_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["--- NEW SESSION STARTED ---", time.strftime("%Y-%m-%d %H:%M:%S")])
    st.session_state["session_started"] = True  # ensures flag is added only once per session

# Category filter
selected_category = st.selectbox("Filter by category:", ["All"] + df["Category"].unique().tolist())

filtered_df = df[df["Category"] == selected_category] if selected_category != "All" else df
filtered_df = filtered_df.reset_index(drop=True)
embeddings_filtered = book_embeddings[filtered_df.index]

# User query input
user_query = st.text_input("Enter the name of book or brief description:")

if user_query:
    # Start time to calculate engagement
    query_start_time = time.time()  

    query_embedding = compute_query_embedding(user_query)
    if query_embedding is not None:
        method = "cosine" if ab_variant == "A" else "dot_product"
        top_books = get_top_recommendations(query_embedding, filtered_df, embeddings_filtered, method=method)

        # Display recommendations 
        st.subheader(f"Recommended Books")
        for _, row in top_books.iterrows():
            book_title = row["Title"]
            st.write(f"**{row['Title']}** by {row['Authors']}")
            st.write(f"Category: {row['Category']}, Publisher: {row['Publisher']}")

            # View button to see book descriptions
            if st.button(f"View {book_title}"):
                log_user_action(user_query, book_title, "View", ab_variant, query_start_time)
                st.write(f"**Description:** {row['Description']}")

            # Buy button to track purchases
            if st.button(f"Buy {book_title}"):
                log_user_action(user_query, book_title, "Buy", ab_variant, query_start_time)
                st.write(f"**Purchased:** {book_title}")
