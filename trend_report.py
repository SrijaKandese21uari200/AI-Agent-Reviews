import pandas as pd

def generate_trend(topics_df, start_date, end_date, rare_threshold=5):
    """
    Generate trend table: rows=topics, cols=dates, values=frequency.
    Rare topics are collapsed into 'Miscellaneous'.
    """
    pivot = pd.pivot_table(
        topics_df,
        index="topic",
        columns="date",
        aggfunc="size",
        fill_value=0
    )
    all_dates = pd.date_range(start_date, end_date).date
    pivot = pivot.reindex(columns=all_dates, fill_value=0)

    topic_totals = pivot.sum(axis=1)
    rare_topics = topic_totals[topic_totals < rare_threshold].index
    if len(rare_topics) > 0:
        misc_row = pivot.loc[rare_topics].sum()
        pivot = pivot.drop(index=rare_topics)
        pivot.loc["Miscellaneous"] = misc_row

    return pivot.sort_index()
