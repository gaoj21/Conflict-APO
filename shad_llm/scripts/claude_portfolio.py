"""Encoder-decoder prompt optimization: the judge = a small portfolio of prompts (encoder) + a per-attribute readout
over all their scores and the item's length (decoder). A prompt edit is worth what it adds to the portfolio, so an
edit that helps one attribute and hurts another is still kept for the attribute it helps; length moves are absorbed by
the decoder and earn nothing.

Each step: refit the decoder (on fit) -> the optimizer sees, per attribute, the items the portfolio ranks most wrongly
(cross-fitted on fit) -> it writes CANDS new prompts -> each candidate is scored on sel by the mean per-attribute
Spearman gain of portfolio + candidate over portfolio, minus COST per extra call (relative tokens); the best candidate
joins if its score is positive (keeping the portfolio scores 0). Same budget as scripts/claude_opt.py arms.

  python scripts/claude_portfolio.py results/claude_hs2 [--steps 4] [--cands 3] [--workers 10]
  python scripts/claude_portfolio.py results/claude_hs2 --test      # judge members on test, evaluate
"""
import itertools, json, os, re, subprocess, sys

import numpy as np, pandas as pd

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1] + [ARGV[0], "portfolio"] + ARGV[1:]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_opt.py")).read()
     .split("# ---------------------------------------------------------------- main loop")[0])
COST = 0.01  # mean-Spearman price of one extra judge call per item (smooth cost term, scaled by relative tokens)
J = [x + "_j" for x in A]
K = len(A)


# ---------------------------------------------------------------- decoder over the portfolio
def table(members, split):
    """One row per item judged by every member: gold, z, and member i's attribute scores as columns i:<attr>_j."""
    out = None
    for i, m in enumerate(members):
        d = m[split][["id", "z"] + A + J].rename(columns={c: f"{i}:{c}" for c in J})
        out = d if out is None else out.merge(d[["id"] + [f"{i}:{c}" for c in J]], on="id")
    return out


def decoder(tr, n, folds=5):
    """Per attribute: ridge from all members' scores + length, shrunk toward an anchor (one member's own score, or
    the average of all members' own scores), anchor and strength chosen by within-fold CV Spearman."""
    cols = [f"{i}:{c}" for i in range(n) for c in J]
    m, s = tr[cols].mean().to_numpy(), tr[cols].std().replace(0, 1).fillna(1).to_numpy()
    zm, zs = tr.z.mean(), tr.z.std()
    feat = lambda d: np.c_[(d[cols].to_numpy(float) - m) / s, (d.z - zm) / zs]
    X, Y = feat(tr), std(tr[A].to_numpy(float))
    fold = np.arange(len(X)) % folds
    W = np.zeros((X.shape[1], K))
    for k in range(K):
        own = [i * K + k for i in range(n)]
        E = [np.eye(X.shape[1])[j] for j in own] + ([np.isin(np.arange(X.shape[1]), own) / n] if n > 1 else [])

        def fit(Xa, ya, e, lam):
            xe = Xa @ e
            w0 = e * (xe @ ya) / max(xe @ xe, 1e-9)
            return w0 if lam is None else np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ ya + lam * w0)

        opts = list(itertools.product(range(len(E)), LAMS + [None]))
        cv = [np.nanmean([sp(Y[fold == f, k], X[fold == f] @ fit(X[fold != f], Y[fold != f, k], E[e], lam))
                          for f in range(folds)]) for e, lam in opts]
        e, lam = opts[int(np.nanargmax(cv))]  # argmax keeps the raw scores' ties (a weighted average breaks them)
        W[:, k] = fit(X, Y[:, k], E[e], lam)
    return lambda d: feat(d) @ W


def sel_score(members):
    tr, te = table(members, "fit"), table(members, "sel")
    return per_task(te[A].to_numpy(float), decoder(tr, len(members))(te)), te


def gain(members, cand):
    """Per-attribute Spearman gain on sel of members + cand over members, on the sel rows all of them judged."""
    new, te = sel_score(members + [cand])
    old_tr, old_te = table(members, "fit"), table(members, "sel")
    old_te = old_te[old_te.id.isin(te.id)]
    old = per_task(old_te[A].to_numpy(float), decoder(old_tr, len(members))(old_te))
    return new - old


def tokens(m):
    return float((m["fit"].in_tok + m["fit"].out_tok).mean()) if "in_tok" in m["fit"] else 1.0


# ---------------------------------------------------------------- feedback: items the portfolio ranks most wrongly
def portfolio_feedback(members):
    d = table(members, "fit")
    reason = members[0]["fit"].set_index("id").reason
    P = np.zeros((len(d), K))
    fold = np.arange(len(d)) % 5
    for f in range(5):
        P[fold == f] = decoder(d[fold != f].reset_index(drop=True), len(members))(d[fold == f])
    H = d[A].to_numpy(float)
    r = np.nanmean([per_task(H[fold == f], P[fold == f]) for f in range(5)], 0)
    pct = lambda x: pd.Series(x).rank(pct=True).to_numpy()
    out = ["Spearman correlation of the current team's combined scores with the human scores on held-out items: " +
           ", ".join(f"{x} {v:.2f}" for x, v in zip(A, r))]
    for k in np.argsort(r):  # weakest attribute first, more examples for weaker ones
        n_ex = 3 if k == np.argmin(r) else 2 if r[k] < np.median(r) else 1
        err = pct(H[:, k]) - pct(P[:, k])
        for i in np.argsort(-np.abs(err))[:n_ex]:
            u = USER[d.id.iloc[i]]
            u = u if len(u) < 1800 else u[:1500] + " [...] " + u[-300:]
            out.append(f"--- {A[k]}: humans put this item in the {'top' if err[i] > 0 else 'bottom'} part "
                       f"(human {A[k]} {H[i, k]:g}; percentile {100 * pct(H[:, k])[i]:.0f}), the team at percentile "
                       f"{100 * pct(P[:, k])[i]:.0f}.\n<item>\n{u}\n</item>\nHuman scores: " +
                       ", ".join(f"{x} {H[i, j]:g}" for j, x in enumerate(A)) +
                       f"\nFirst version's reasoning (end): {reason.get(d.id.iloc[i], '')}\n")
    return "\n".join(out), dict(decoder_sel_fit=[float(x) for x in r])


PORT_NOTE = """The judge is a TEAM of instruction versions: every item is judged once with each version, and a per-attribute calibration layer combines all versions' scores (and the item's length) into the final score of each attribute, using each version only for the attributes it helps. It also already corrects leniency, how much each attribute should reward length, and how attributes leak into each other, so edits that only move scores along those directions add nothing.
Your new version is ADDED to the team. It does not have to be good at every attribute: it helps if it captures what the team currently gets wrong (the items below), even if it is worse than the others on attributes the team already rates well. Each version costs one more judge call per item, so it has to add something real."""

PORT_TASK = """<team>
{team}
</team>

<feedback>
{feedback}
</feedback>

Write {n} different new versions of the instructions, each a complete set of instructions for the judge (start from any team member's instructions), each trying a different idea about what the team misses. Reply with a one-line rationale for each, then the versions as <prompt1>...</prompt1>, <prompt2>...</prompt2>, ..."""


def propose(members, step):
    path = f"{O}/prompts/s{step}_opt.txt"
    fb, info = portfolio_feedback(members)
    system = OPT_SYS.format(attrs=", ".join(A), lo=LO, hi=HI) + "\n\n" + PORT_NOTE
    team = "\n\n".join(f"<version name=\"{m['name']}\">\n{m['prompt'].strip()}\n</version>" for m in members)
    user = PORT_TASK.format(team=team, feedback=fb, n=a.cands)
    if os.path.exists(path):
        txt = open(path).read()
    else:
        txt = ""
        for _ in range(3):
            p = subprocess.run(["claude", "-p", "--model", a.optimizer, "--output-format", "json", "--no-session-persistence",
                                "--tools", "", "--setting-sources", "", "--system-prompt", system],
                               input=user, capture_output=True, text=True, timeout=900, cwd="/tmp")
            try:
                o = json.loads(p.stdout)
            except Exception:  # noqa: BLE001
                continue
            if o.get("is_error"):
                raise Abort("optimizer: " + str(o.get("result", ""))[:300])
            txt = o.get("result", "")
            log(f"  optimizer ${o.get('total_cost_usd', 0):.3f}")
            if "<prompt1>" in txt:
                break
        open(path, "w").write(txt)
        open(f"{O}/prompts/s{step}_opt_input.txt", "w").write(system + "\n\n=====\n\n" + user)
    cands = [c.strip() for c in re.findall(r"<prompt\d>(.*?)</prompt\d>", txt, re.S)][:a.cands]
    if not cands:
        raise Abort(f"optimizer reply without prompts at step {step}")
    return cands, info


def member(name, prompt):
    return dict(name=name, prompt=prompt, fit=frame(judge_rows(name, prompt, "fit"), "fit"),
                sel=frame(judge_rows(name, prompt, "sel"), "sel"))


def run():
    v0 = open(f"{R}/prompts/v0.txt").read()
    members = [member("v0", v0)]
    t0 = tokens(members[0])
    s, _ = sel_score(members)
    log(f"[portfolio] step 0 sel {np.round(s, 3)} mean {np.nanmean(s):.3f}")
    for step in range(1, a.steps + 1):
        cands, info = propose(members, step)
        rows = []
        for k, c in enumerate(cands):
            name = f"s{step}c{k}"
            open(f"{O}/prompts/{name}.txt", "w").write(c)
            m = member(name, c)
            g = gain(members, m)
            score = float(np.nanmean(g)) - COST * tokens(m) / t0
            rows.append((score, m, dict(name=name, score=score, gain=g.round(4).tolist())))
            log(f"  {name}: score {score:+.3f} gain {np.round(g, 3)}")
        best = max(rows, key=lambda r: r[0])
        added = best[0] > 0
        if added:
            members = members + [best[1]]
        s, _ = sel_score(members)
        open(f"{O}/log.jsonl", "a").write(json.dumps(dict(step=step, added=best[2]["name"] if added else None,
                                                          members=[m["name"] for m in members], feedback=info,
                                                          cands=[r[2] for r in rows], sel=s.round(4).tolist())) + "\n")
        log(f"[portfolio] step {step}: {'add ' + best[2]['name'] if added else 'keep'}; members "
            f"{[m['name'] for m in members]} sel {np.round(s, 3)} mean {np.nanmean(s):.3f}")
    open(f"{O}/final.txt", "w").write(",".join(m["name"] for m in members))


def test_eval():
    names = open(f"{O}/final.txt").read().strip().split(",")
    members = []
    for nm in names:
        pr = open(f"{R}/prompts/v0.txt" if nm == "v0" else f"{O}/prompts/{nm}.txt").read()
        m = member(nm, pr)
        m["test"] = frame(judge_rows(nm, pr, "test"), "test")
        members.append(m)
    tr = pd.concat([table(members, "fit"), table(members, "sel")], ignore_index=True)
    te = table(members, "test")
    H = te[A].to_numpy(float)
    print(f"=== {os.path.basename(R)} portfolio {names} test n={len(te)}: {' '.join(x[:6] for x in A)} | mean")
    for n in range(1, len(members) + 1):
        r = per_task(H, decoder(tr, n)(te))
        print(f"  first {n} member(s) ({n} calls)  " + " ".join(f"{x:6.3f}" for x in r) + f" | {np.nanmean(r):.3f}")


if __name__ == "__main__":
    try:
        test_eval() if a.test else run()
    except Abort as e:
        log(f"ABORT {e}")
        sys.exit(2)
