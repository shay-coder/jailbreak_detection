"""Rebuilds data/prompts.csv from the In-The-Wild Jailbreak Prompts dataset
(Shen et al., ACM CCS 2024). All jailbreak prompts are kept and an equal number
of regular prompts are sampled to form a balanced dataset."""

import os
import subprocess
import pandas as pd

REPO_URL = "https://github.com/verazuo/jailbreak_llms.git"
CLONE_DIR = "jailbreak_llms_source"


def ensure_source():
    if not os.path.isdir(CLONE_DIR):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR],
                       check=True)


def build():
    base = os.path.join(CLONE_DIR, "data", "prompts")
    jb = pd.read_csv(os.path.join(base, "jailbreak_prompts_2023_12_25.csv"))
    reg = pd.read_csv(os.path.join(base, "regular_prompts_2023_12_25.csv"))

    jb_df = jb[["prompt"]].copy()
    jb_df["label"] = 1
    reg_df = reg[["prompt"]].sample(n=len(jb_df), random_state=42).copy()
    reg_df["label"] = 0

    data = pd.concat([jb_df, reg_df], ignore_index=True)
    data["prompt"] = (data["prompt"].astype(str)
                      .str.replace(r"\s+", " ", regex=True).str.strip())
    data = data[data["prompt"].str.len() > 0].drop_duplicates(subset=["prompt"])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    data.to_csv("data/prompts.csv", index=False)
    print(f"Wrote data/prompts.csv: {len(data)} prompts "
          f"({int((data.label == 1).sum())} jailbreak, "
          f"{int((data.label == 0).sum())} regular)")


if __name__ == "__main__":
    ensure_source()
    build()
