"""PPELen splits: PPE human-preference pairs (HyperAgents domains/ppe, 100-word-filtered subsets) where the human
preferred the LONGER response (kind=longer) or the SHORTER one (kind=shorter), balanced. A judge's length preference
trades the two accuracies against each other (the spurious length axis). A strict content screen (as HelpSteer2SafeG,
plus minors) drops sensitive pairs; nothing is rewritten. Train = PPE train + val, test = PPE test (balanced).

  python scripts/make_ppelen.py
"""

import os
import re

import pandas as pd

SRC = os.path.expanduser("~/Desktop/hyperagents/domains/ppe/dataset_filtered_100_{}.csv")
WORDS = r"""china|chinese|beijing|taiwan|tibet|xinjiang|uyghur|hong kong|tiananmen|communis|ccp|xi jinping|\bmao\b|falun|
dalai|party|government|politic|president|election|democra|protest|revolution|regime|dictator|censor|human rights|
propaganda|military|army|\bwar\b|\bwars\b|soldier|terror|nazi|hitler|genocide|holocaust|israel|palestin|gaza|ukrain|
russia|putin|trump|biden|obama|religio|\bgod\b|jesus|islam|muslim|christian|bible|quran|church|abortion|\bgay\b|lesbian|
lgbt|transgender|racis|slave|\bsex|porn|nude|naked|erotic|kill|murder|suicide|death|\bdead\b|\bdie\b|blood|\bgun|weapon|
bomb|drug|cocaine|marijuana|alcohol|gambl|casino|crime|criminal|prison|police|steal|fraud|hack|violen|abuse|rape|assault|
torture|\bhate|kidnap|\bweed\b|cannabis|vaccine|covid|disease|cancer|medical|medicine|doctor|patient|surgery|diagnos|
symptom|pregnan|year-old|years old|\bteen|underage|\bminor\b|\bchild|\bkid\b|\bkids\b|\bgirl|\bboy\b|lingerie|breast|
kiss|seduc|fetish|bdsm|intimate|horny|\bhot\b|sexy""".replace("\n", "")
PAT = re.compile(WORDS, re.I)


def screen(df):
    txt = df.prompt.astype(str) + " " + df.response_1.astype(str) + " " + df.response_2.astype(str)
    return df[~txt.map(lambda s: bool(PAT.search(s)))]


os.makedirs("data/PPELen", exist_ok=True)
tr = screen(pd.concat([pd.read_csv(SRC.format("train")), pd.read_csv(SRC.format("val"))]))
te = screen(pd.read_csv(SRC.format("test")))
for name, d in (("train", tr), ("test", te)):
    n = d.kind.value_counts().min()
    parts = [g.sample(n=n, random_state=0).reset_index(drop=True) for _, g in d.groupby("kind")]
    d = pd.concat(parts).sort_index(kind="stable")  # kinds interleaved, so every head(n) is balanced
    d = d.rename(columns={"best": "preference"}).reset_index(drop=True)
    d = d[["kind", "prompt", "response_1", "response_2", "preference", "source_domain"]]
    d.to_parquet(f"data/PPELen/PPELen-{name}.parquet")
    print(name, len(d), d.kind.value_counts().to_dict(), "pref", d.preference.value_counts().to_dict())
