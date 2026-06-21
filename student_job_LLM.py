from datetime import datetime
from glob import glob
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from langchain_openai import ChatOpenAI

INPUT_FILES = ["reddit_comments.csv"]
MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

VLLM_ENDPOINT = "http://127.0.0.1:8000/v1"
API_KEY = "password"
LIMIT = int(5_000)  # Maximum is ~100_000


def read_data(input_files: list, limit: int, seed: int = 32) -> pd.DataFrame:
    """
    Reads data from input_files with a read_csv function in python

    Args:
        input_files: list of paths to the input files.

    Return:
        Returns a pandas dataframe.
    """
    dfs = []
    for input_file in input_files:
        df = pd.read_csv(input_file).sample(n=limit, replace=False, random_state=seed)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def analyze_comment(comment: str) -> dict:
    try:
        result = structured_llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT_STOCK},
            {"role": "user", "content": comment}
        ])
        return {"tickers": result.tickers, "sentiment": result.sentiment.value}
    except Exception as e:
        return {"tickers": ["NA"], "sentiment": "neutral"}

class SentimentEnum(str, Enum):
        very_positive = "very positive"
        positive = "positive"
        neutral = "neutral"
        negative = "negative"
        very_negative = "very negative"

class StockComment(BaseModel):
    tickers: list[str]         
    sentiment: SentimentEnum

if __name__ == "__main__":
    # Wait the llm becomes available. A while loop with a network request and check over the
    # status would be better, but for sake of simplicity we will use a sleep.

    t0 = datetime.now()

    # OpenAI client pointing to our local model
    client = OpenAI(base_url=VLLM_ENDPOINT, api_key=API_KEY)

    # Read data files
    df = read_data(INPUT_FILES, LIMIT)

    # Instantiate the LangChain OpenAI client
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=VLLM_ENDPOINT,
        api_key=API_KEY,
        max_tokens=500
    )



    SYSTEM_PROMPT_STOCK = """You are a financial analyst assistant.
Given a Reddit comment, extract:
1. The stock ticker symbol(s) of the companie(s) being discussed (e.g. AAPL for Apple, TSLA for Tesla).
   - If the comment does not mention any publicly traded company, return ["NA"].
   - If multiple companies are mentioned, return all of them in the list.
2. The overall sentiment of the comment toward those stocks. (Assume the sentiment applies to all extracted stocks).

Respond strictly in the requested JSON format. No explanations."""

    structured_llm = llm.with_structured_output(StockComment)
    
    results = [None] * len(df)
    
    print(f"Starting sentiment analysis on {len(df)} comments using 32 threads...")
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(analyze_comment, row["comments"]): i 
                   for i, row in df.iterrows()}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    df["tickers"] = [r["tickers"] for r in results]
    df["sentiment"] = [r["sentiment"] for r in results]

    # Explode the dataframe so each ticker gets its own row, then rename
    df = df.explode("tickers").rename(columns={"tickers": "ticker"})

    # Filter out comments that have no recognized ticker ("NA")
    df = df[df["ticker"] != "NA"]

    print("\n--- Sample of extracted results ---")
    print(df[["comments", "ticker", "sentiment"]].head(10))

    # Save to CSV so the Agent can read it
    output_csv = "sentiment_analysis_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults successfully saved to {output_csv}")
    
    t1 = datetime.now()
    print(f"Total execution time: {t1 - t0}")
