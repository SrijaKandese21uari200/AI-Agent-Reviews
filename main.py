import os
import pandas as pd
from datetime import datetime, timedelta

from src.fetch_reviews import fetch_all_reviews
from src.preprocess import clean_text
from src.topic_agent import extract_cluster_labels
from src.trend_report import generate_trend
from src.utils import cluster_reviews

APP_ID = "in.swiggy.android"
END_DATE = datetime.today().date()
START_DATE = datetime(2024, 6, 1).date()
OUTPUT_FILE = "output/trend_report.csv"

def main():
    print("=== AI Agent: Review Trend Analysis (Groq, Optimized for Tokens) ===")

    # Step 1: Fetch reviews
    df = fetch_all_reviews(app_id=APP_ID, max_count=50000)
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    print(f"Fetched {len(df)} reviews total.")

    # Step 2: Preprocess
    df["clean_text"] = df["content"].apply(clean_text)

    # Step 3: Cluster reviews (reduce Groq calls)
    cluster_assignments, reps = cluster_reviews(df["clean_text"].tolist(), similarity_threshold=0.85)
    rep_texts = [df["clean_text"].iloc[i] for i in reps]
    print(f"Clustered into {len(reps)} groups (will call Groq that many times).")

    # Step 4: Groq labeling for cluster reps
    rep_labels = extract_cluster_labels(rep_texts, batch_size=20)

    # Step 5: Assign labels back to all reviews
    cluster_to_label = {cid: label for cid, label in enumerate(rep_labels)}
    df["topic"] = [cluster_to_label[cid] for cid in cluster_assignments]

    # Step 6: Generate trend report (last 30 days window)
    analysis_start = END_DATE - timedelta(days=30)
    trend_df = generate_trend(df[["date", "topic"]], analysis_start, END_DATE)

    # Step 7: Save
    os.makedirs("output", exist_ok=True)
    trend_df.to_csv(OUTPUT_FILE)
    print(f"✅ Trend report saved at {OUTPUT_FILE}")
    print(trend_df.head(10))

if __name__ == "__main__":
    main()
