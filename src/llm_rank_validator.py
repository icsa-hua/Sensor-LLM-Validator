import pandas as pd
import subprocess
import re

MODEL_NAME = "gemma3:12b"

def query_ollama(prompt: str) -> str:
    """
    Sends a prompt to Ollama and returns the raw model text output.
    """
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()


def extract_rank_and_reason(response: str):
    """
    Extracts:
      - rank (1–10)
      - reason (string)
    from the LLM response.

    Expected response format:
       RANK: X
       REASON: some explanation...
    """

    # Rank extraction (first number 1–10 found)
    rank_match = re.search(r"\b([1-9]|10)\b", response)
    rank = int(rank_match.group(1)) if rank_match else 5

    # Reason extraction
    reason_match = re.search(r"REASON:(.*)", response, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # fallback: remove the rank number and treat the rest as reason
        reason = re.sub(r"\b([1-9]|10)\b", "", response).strip()

    return rank, reason


def build_prompt(row) -> str:
    """
    Builds the scoring + reasoning prompt for a single sensor sample.
    """

    return f"""
You are a specialized sensor-data validation LLM trained to assess whether a
sensor reading is realistic, physically consistent, and statistically plausible.

You will evaluate the trustworthiness of each sensor sample using the following
environmental and degradation-related factors:

- Scenario: expected harshness level (mild / moderate / harsh) and its typical environmental ranges
- Temperature: operational temperature in °C and how it aligns with expected voltage behavior
- Humidity: relative humidity and its effect on drift, noise, and degradation
- Month: seasonal context that influences temperature/humidity realism
- Burst probability: likelihood of burst noise events due to device aging
- Thermal cycles: accumulated thermal stress affecting offset and drift
- Output voltage: the final measured output you must judge for realism given the above conditions

TASK:
1. Give a **validity score from 1 to 10** (1 = invalid, 10 = highly valid)
2. Explain briefly *why* you chose that score (1–2 sentences)

RESPONSE FORMAT (must follow exactly):
RANK: <number 1–10>
REASON: <short explanation>

Here is the sample:

Scenario: {row['scenario']}
Temperature (C): {row['temperature']}
Humidity (%): {row['humidity']}
Month: {row['month']}

Burst Probability: {row['burst_probability']}
Thermal Cycles: {row['thermal_cycles']}
Output Voltage (V): {row['output_voltage']}
"""


def rank_csv(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    ranks = []
    reasons = []

    for _, row in df.iterrows():
        prompt = build_prompt(row)
        response = query_ollama(prompt)
        rank, reason = extract_rank_and_reason(response)

        print(row)
        print(rank)
        print(reason)

        ranks.append(rank)
        reasons.append(reason)

    df["llm_rank"] = ranks
    df["llm_reason"] = reasons

    df.to_csv(output_path, index=False)
    print(f"✔ Wrote ranked CSV with explanations to {output_path}")


if __name__ == "__main__":
    rank_csv("sensor_created.csv", "sensor_ranked.csv")
