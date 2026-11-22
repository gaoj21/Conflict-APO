"""TL;DR posts for RL (trl-lib/tldr: the Reddit TL;DR data of Stiennon et al. 2020, filtered).

train: prompts for GRPO (the train split); eval: a fixed set of held-out posts (the test split) with the human-written
reference TL;DR, used for the gold comparison. None of these posts is in the human axis ratings the decoder was fitted
on (those come from the TL;DR validation posts)."""
import re

from datasets import load_dataset

INSTRUCTION = "Write a TL;DR of this Reddit post: one or two sentences that capture its main point."


def parse(prompt):
    m = re.match(r"SUBREDDIT: (.*?)\n\nTITLE: (.*?)\n\nPOST: (.*)\n\nTL;DR:\s*$", prompt, re.S)
    return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip()) if m else None


def rows(split, n, seed=0):
    ds = load_dataset("trl-lib/tldr", split=split).shuffle(seed=seed)
    out = []
    for r in ds:
        p = parse(r["prompt"])
        if p is None:
            continue
        sub, title, post = p
        out.append(dict(title=title, post=post, subreddit=sub, reference=r["completion"].strip(),
                        prompt=[dict(role="user", content=f"{INSTRUCTION}\n\nSubreddit: {sub}\nTitle: {title}\n\n{post}")]))
        if len(out) >= n:
            break
    return out
