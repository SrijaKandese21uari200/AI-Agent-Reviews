from groq import Groq
import os, time, random, groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def safe_groq_request(prompt, retries=5):
    """Retry wrapper for Groq to handle rate limits."""
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
        except groq.RateLimitError:
            wait = (2 ** attempt) + random.random()
            print(f"Rate limit hit. Waiting {wait:.2f}s before retry...")
            time.sleep(wait)
    raise Exception("Groq request failed after retries.")

def extract_cluster_labels(texts, batch_size=20):
    """
    Get normalized topic labels for a small list of cluster reps.
    Returns one label per input text.
    """
    topics = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        joined = "\n".join([f"- {t}" for t in batch if t.strip()])

        prompt = f"""
        You are analyzing food delivery app reviews.
        Task:
        - Extract the main issue/request/feedback topic for EACH review.
        - Normalize similar ideas into ONE consistent label.
        - Use short labels (2–4 words).
        - Examples: "Delivery issue", "Food stale", "App crash".
        - If unclear, return "Miscellaneous".

        Reviews:
        {joined}

        Return ONLY one topic per review, same order, one per line.
        """

        response = safe_groq_request(prompt)
        out = response.choices[0].message.content.strip().split("\n")
        out = [o.strip("-• ").strip() for o in out if o.strip()]

        while len(out) < len(batch):
            out.append("Miscellaneous")
        out = out[:len(batch)]
        topics.extend(out)

    return topics
