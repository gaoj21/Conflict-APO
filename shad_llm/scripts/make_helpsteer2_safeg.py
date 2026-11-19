"""HelpSteer2SafeG splits: HelpSteer2Safe with a second, stricter content screen for hosted models with content-safety
moderation (glm-5.3-flash on juapi rejected rows of HelpSteer2Safe). Every prompt whose prompt or any response mentions
politics, religion, war, violence, crime, sex, drugs, health/medicine or similar topics is dropped with all its
responses. Avoids sensitive content; nothing is rewritten.

  python scripts/make_helpsteer2_safeg.py
"""

import os
import re

import pandas as pd

WORDS = r"""china|chinese|beijing|taiwan|tibet|xinjiang|uyghur|hong kong|tiananmen|communis|ccp|xi jinping|mao|falun|
dalai|party|government|politic|president|election|democra|protest|revolution|regime|dictator|censor|human rights|
propaganda|military|army|war\b|wars\b|soldier|terror|nazi|hitler|genocide|holocaust|israel|palestin|gaza|ukrain|russia|
putin|trump|biden|obama|religio|god\b|jesus|islam|muslim|christian|bible|quran|church|abortion|gay|lesbian|lgbt|
transgender|racis|slave|sex|porn|nude|naked|erotic|kill|murder|suicide|death|dead\b|die\b|blood|gun|weapon|bomb|drug|
cocaine|marijuana|alcohol|gambl|casino|crime|criminal|prison|police|steal|fraud|hack|violen|abuse|rape|assault|torture|
hate|racist|kidnap|weed|cannabis|vaccine|covid|disease|cancer|medical|medicine|doctor|patient|surgery|diagnos|symptom|
pregnan""".replace("\n", "")
PAT = re.compile(WORDS, re.I)
SRC, DST = "data/HelpSteer2Safe/HelpSteer2Safe-{}.parquet", "data/HelpSteer2SafeG/HelpSteer2SafeG-{}.parquet"

os.makedirs("data/HelpSteer2SafeG", exist_ok=True)
for split in ("train", "test"):
    df = pd.read_parquet(SRC.format(split))
    hit = (df.prompt.astype(str) + " " + df.response.astype(str)).map(lambda s: bool(PAT.search(s)))
    keep = df[~df.prompt.isin(set(df.prompt[hit]))].reset_index(drop=True)
    keep.to_parquet(DST.format(split))
    print(f"{split}: {len(df)} -> {len(keep)} rows ({keep.prompt.nunique()} prompts)")
