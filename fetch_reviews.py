from google_play_scraper import reviews, Sort
import pandas as pd

def fetch_all_reviews(app_id="in.swiggy.android", max_count=50000):
    """
    Fetch all reviews (up to max_count).
    Default reduced to 50k for speed and token saving.
    """
    result, _ = reviews(
        app_id,
        lang="en",
        country="in",
        sort=Sort.NEWEST,
        count=max_count
    )
    df = pd.DataFrame(result)
    df["date"] = pd.to_datetime(df["at"]).dt.date
    return df[["userName", "score", "content", "date"]]
