"""Reproduce decomposition modes (SSS/SSC/SCC/CCC) + single-task (ST) on a local endpoint.

Mirrors scripts/run_decomposition_modes.py, but uses the current runner API
(the original script references the removed create_shared_limits()).

Usage:
    python scripts/run_local.py --modes SSS SSC SCC CCC ST --val mae --seeds 1 2 3 --steps 12
"""

import argparse
import os
import shutil
import sys
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import litellm  # noqa: E402

# aiohttp transport keeps idle keep-alive sockets open per request and exhausts local ports at high
# request volume; httpx has a bounded pool. Module level so spawned workers pick it up too.
litellm.disable_aiohttp_transport = True
# concurry's default SlidingWindow limiter stores one timestamp per token and rebuilds the list on every acquire:
# with ~1k-token prompts its 60 s window holds millions of entries and pins each worker at 100% CPU. GCRA is O(1).
from concurry.config import global_config as _concurry_config  # noqa: E402
from concurry.core.constants import RateLimitAlgorithm as _RLA  # noqa: E402

_concurry_config.defaults.rate_limit_algorithm = _RLA.GCRA

from when_gradients_collide.algorithm import TextGradAF, TextGradDRO, TextGradSAI, TextGradSAIP  # noqa: E402
from when_gradients_collide.algorithm import CTG, MRB, MRB2, MRB6, MRB7, MRB8, MRBS, MRBS2, MRBS3, MRBNG, MRBR, RSI, RSI2, TextGrad  # noqa: E402
from when_gradients_collide.config import wgc_config  # noqa: E402
from when_gradients_collide.experiment_config import LLMConfig  # noqa: E402
from when_gradients_collide.expt import dataset as ds_module  # noqa: E402
from when_gradients_collide.expt.runner import (  # noqa: E402
    create_gradient_llm,
    create_loss_llm,
    create_optimizer_llm,
    create_task_llm,
    get_initial_prompt,
)

# Pre-import lazily-loaded async backends in the main thread: with many jobs
# starting concurrently, per-worker event-loop threads otherwise deadlock on
# the import lock (observed in anyio.get_async_backend during openai retries).
import anyio._backends._asyncio  # noqa: E402,F401
import httpx  # noqa: E402,F401
import litellm  # noqa: E402,F401
import openai  # noqa: E402,F401

MODE_CONFIGS = {
    "SSS": ("separate_tasks", "separate_tasks", "separate_tasks"),
    "SSC": ("separate_tasks", "separate_tasks", "combine_all_tasks"),
    "SCC": ("separate_tasks", "combine_all_tasks", "combine_all_tasks"),
    "CCC": ("combine_all_tasks", "combine_all_tasks", "combine_all_tasks"),
}
# MRB variants: name -> (algorithm class, extra constructor kwargs, decomposition mode).
# MRB2 (joint): per-metric loss, one gradient + one optimizer over all metrics (SCC structure).
# MRB2 as first run (no rule hygiene); MRB2b adds the hygiene defaults of mrb_v2.MRB2.
MRB2_LEGACY = {"improve_worst": 0.05, "improve_cross_mean": None, "guard_harm_delta": 0.05,
               "max_rule_words": None, "dedup": False}
# MRB2c: guards as learned constraints with dual-ascent weights; novelty ignores guard citations;
# improvement rules retire only after >= 2 harmful observations.
MRB2C = {"novelty_ignores_guards": True, "guard_dual": True, "improve_retire_min": 2}
# MRB2d: MRB2c + noise-calibrated thresholds (paired bootstrap on the attribution rows) + failed-edit memory.
MRB2D = {**MRB2C, "noise_calibrated": True, "failed_memory": 6}
# MRB3: MRB2d + multi-task-loss constraint machinery: conflict matrix (GradVac), Nash-bargaining candidate
# selection (Nash-MTL), progress weights (GradNorm), uncertainty weights (Kendall); ablations below incl. RLW.
MRB3 = {**MRB2D, "conflict_matrix": True, "nash_select": True, "progress_weights": True,
        "uncertainty_weights": True, "separate_attr_rows": True}
MRB3MO = {**MRB3, "consolidate": False, "harm_delta": 1.0, "gain_delta": 0.6, "guard_harm_delta": 1.6,
          "improve_worst": 2.0, "rewrite_candidate": True}
# MRB3-MO2: fresh current-prompt eval each step (no winner's curse) + looser Nash veto (Δ <= -2.7 units).
MRB3MO2 = {**MRB3MO, "fresh_base": True, "nash_scale_min": 3.0}
# MRB4: MRB3-MO2 scaled up: 3 pooled optimizer proposals per step, successive-halving candidate screening
# (150 of the 500 selection rows; the best 5 get the rest), early stop after 2 steps without an update.
MRB4 = {**MRB3MO2, "proposal_samples": 3, "race_rows": 150, "race_keep": 5, "patience": 2}
# MRB5: MRB3-MO2 + output analysis for the optimizer: error examples show the response, the judge's reasoning
# and all scores; error modes ranked by count x |error|; aggregate judge-vs-human stats; early stop.
MRB5 = {**MRB3MO2, "rich_examples": True, "mode_rank": "mass", "output_stats": True, "patience": 2}
MRB_VARIANTS = {
    "MRB": (MRB, {}, "SSS"),
    "MRB-noguard": (MRB, {"use_guards": False}, "SSS"),
    "MRB-nopersist": (MRB, {"persist_rules": False}, "SSS"),
    "MRB2": (MRB2, dict(MRB2_LEGACY), "SCC"),
    "MRB2-jointattr": (MRB2, {**MRB2_LEGACY, "attribution_mode": "joint"}, "SCC"),
    "MRB2-noguard": (MRB2, {**MRB2_LEGACY, "use_guards": False}, "SCC"),
    "MRB2-sep": (MRB2, {**MRB2_LEGACY, "joint": False}, "SSS"),
    "MRB2b": (MRB2, {}, "SCC"),
    "MRB2b-sep": (MRB2, {"joint": False}, "SSS"),
    "MRB2c": (MRB2, dict(MRB2C), "SCC"),
    "MRB2c-sep": (MRB2, {**MRB2C, "joint": False}, "SSS"),
    "MRB2d": (MRB2, dict(MRB2D), "SCC"),
    "MRB2d-sep": (MRB2, {**MRB2D, "joint": False}, "SSS"),
    "MRB3": (MRB2, dict(MRB3), "SCC"),
    "MRB3-noC": (MRB2, {**MRB3, "conflict_matrix": False}, "SCC"),
    "MRB3-nonash": (MRB2, {**MRB3, "nash_select": False}, "SCC"),
    "MRB3-noprog": (MRB2, {**MRB3, "progress_weights": False}, "SCC"),
    "MRB3-nounc": (MRB2, {**MRB3, "uncertainty_weights": False}, "SCC"),
    "MRB3-rlw": (MRB2, {**MRB3, "progress_weights": False, "uncertainty_weights": False, "random_weights": True}, "SCC"),
    # Multi-objective (val=multi): thresholds in normalized units (1 unit = 0.05 Δρ for quality objectives);
    # consolidation off (the Nash selection already rejects edits that hurt any objective).
    "MRB3-MO": (MRB2, dict(MRB3MO), "SCC"),
    "MRB3-MO2": (MRB2, dict(MRB3MO2), "SCC"),
    "MRB4": (MRB2, dict(MRB4), "SCC"),
    "MRB4-noprop": (MRB2, {**MRB4, "proposal_samples": 1}, "SCC"),
    "MRB4-norace": (MRB2, {**MRB4, "race_rows": 0}, "SCC"),
    "MRB3-MO2-ES": (MRB2, {**MRB3MO2, "patience": 2}, "SCC"),
    "MRB3-MO2-nonash": (MRB2, {**MRB3MO2, "nash_select": False}, "SCC"),
    "MRB3-MO2-norewrite": (MRB2, {**MRB3MO2, "rewrite_candidate": False}, "SCC"),
    "MRB3-MO2-noC": (MRB2, {**MRB3MO2, "conflict_matrix": False}, "SCC"),
    # MRB6: MRB3-MO2 selection + rules as hypotheses induced from ALL candidates, tested blind on later steps.
    "MRB6": (MRB6, dict(MRB3MO2), "SCC"),
    "MRB6-noprune": (MRB6, {**MRB3MO2, "prune": False}, "SCC"),
    "MRB6-chosen": (MRB6, {**MRB3MO2, "learn_from_all": False}, "SCC"),
    # MRB7: MRB6 without pruning + rule-driven proposals (2nd optimizer call) + fresh (unbiased) validation gate.
    "MRB7": (MRB7, dict(MRB3MO2), "SCC"),
    "MRB7-gate": (MRB7, {**MRB3MO2, "rule_proposals": False, "proposal_samples": 1}, "SCC"),
    "MRB7-gen": (MRB7, {**MRB3MO2, "fresh_gate": False}, "SCC"),
    # MRB8: per-metric rules (dataset-level evidence) aggregated into one rulebook + holistic revision candidate.
    "MRB8": (MRB8, dict(MRB3MO2), "SCC"),
    "MRB8-noagg": (MRB8, {**MRB3MO2, "aggregate": False}, "SCC"),
    "MRB8-noholistic": (MRB8, {**MRB3MO2, "holistic": False}, "SCC"),
    # MRB-S: simplified MRB (one edit per metric on 300 rows, merge + fresh gate, per-metric rules aggregated).
    "MRBS": (MRBS, {**MRB3MO2, "rewrite_candidate": False}, "SCC"),
    # MRB-S2: each edit tested only on the task types its metric covers (rows_per_kind per type).
    "MRBS2": (MRBS2, {**MRB3MO2, "rewrite_candidate": False}, "SCC"),
    # MRB-S3: MRB-S2 + gate scores each row once (run with a larger --gate-samples), instability on 300 rows.
    "MRBS3": (MRBS3, {**MRB3MO2, "rewrite_candidate": False}, "SCC"),
    # MRB-NG: no gates (no validation gate, no fresh re-score, no veto); positive-effect edits merged and kept.
    "MRBNG": (MRBNG, {**MRB3MO2, "rewrite_candidate": False}, "SCC"),
    # MRB-NG-all: every proposed edit is applied; the counterfactual test only feeds the rules.
    "MRBNG-all": (MRBNG, {**MRB3MO2, "rewrite_candidate": False, "merge_threshold": float("-inf")}, "SCC"),
    # MRB-R: MRB-NG with per-metric rules rewritten each step from the whole edit history (no statuses / matching).
    "MRBR": (MRBR, {**MRB3MO2, "rewrite_candidate": False}, "SCC"),
    # RSI: self-referential agent (judge + meta instructions), DGM archive, scalar score (DGM-Hyperagents in prompt form).
    "RSI": (RSI, {**MRB3MO2, "rewrite_candidate": False, "children_per_step": 4}, "SCC"),
    # RSI-C: the same loop + per-instruction counterfactual tests (keep positive changes) + per-metric rulebook.
    # RSI2: single-instruction judge (RewardBench 2), DGM archive + self-referential meta instructions; baseline.
    "RSI2": (RSI2, {"children_per_step": 4}, "SCC"),
    # RSI2-R: the same loop + self-evolving rulebook (cited rules, flips, conflict repair, add/delete rule updates).
    "RSI2-R": (RSI2, {"rulebook": True, "children_per_step": 3}, "SCC"),
    "RSI-C": (RSI, {**MRB3MO2, "rewrite_candidate": False, "conflict_aware": True, "children_per_step": 2}, "SCC"),
    "MRB5": (MRB2, dict(MRB5), "SCC"),
    "MRB5-nostats": (MRB2, {**MRB5, "output_stats": False}, "SCC"),
    "MRB3-MO-noC": (MRB2, {**MRB3MO, "conflict_matrix": False}, "SCC"),
    "MRB3-MO-nonash": (MRB2, {**MRB3MO, "nash_select": False}, "SCC"),
    "MRB3-MO-noprog": (MRB2, {**MRB3MO, "progress_weights": False}, "SCC"),
    "MRB3-MO-nounc": (MRB2, {**MRB3MO, "uncertainty_weights": False}, "SCC"),
    "MRB3-MO-norewrite": (MRB2, {**MRB3MO, "rewrite_candidate": False}, "SCC"),
    "MRB3-MO-rlw": (MRB2, {**MRB3MO, "progress_weights": False, "uncertainty_weights": False, "random_weights": True}, "SCC"),
}
# CTG: CCC + constraints (Lagrangian metric constraints, learned rules, trust region, memory pre-check); ablations.
CTG_VARIANTS = {
    "CTG": {},
    "CTG-noL": {"lagrangian": False},
    "CTG-noR": {"rules": False},
    "CTG-noT": {"trust_region": False},
    "CTG-noM": {"memory_check": False},
}
# Conflict remedies carried over from the RSI study (algorithm/textgrad_dro.py), all on the CCC structure:
# DRO = worst-task off-by-one gate (Group DRO); DROS = DRO + surgery note (last step's per-task fixes/breaks);
# CAL = per-task isotonic calibration of the scores (gate on calibrated MAE; tests store cal_<task>).
DRO_VARIANTS = {
    "DRO": {"worst_gate": True},
    "DROS": {"worst_gate": True, "surgery": True},
    "CAL": {"calibrate": True},
}
# Axis-factored TextGrad (algorithm/textgrad_af.py), CCC structure, run with --val none (no gate):
# AF = per-task calibration heads + residual feedback; AFM = plain CCC with the heads only measured (baseline +
# measurement-only calibration control in one run).
AF_VARIANTS = {"AF": {"factor": True}, "AFM": {"factor": False}}
SAI_VARIANTS = {"SAI": {"factor": True}}
SAIP_VARIANTS = {"SAIP": {"factor": True}}
VAL_SHORT = {"off_by_one": "ob1", "mae": "mae", "spearman_correlation": "spearman", "multi": "multi", "none": "none"}
COMMON_PARAMS = {
    "validation_gate_samples": 100,
    "gradient_llm_temperature": 0.3,
    "optimizer_llm_temperature": 0.7,
    "loss_llm_temperature": 0.3,
}


def run_single(*, job_name, mode, val_metric, task_subset, llm_config, dataset_cls, steps, batch_size, out_root, mrb_kwargs=None,
               gate_samples=100, eval_subset=None, attr_repeats=None, objectives=None, task_max_tokens=None,
               eval_every=1):
    dataset = dataset_cls(data_dir=str(REPO / "data"))
    tasks = [t for t in dataset.tasks if task_subset is None or t.task_name in task_subset]
    algo_cls, extra = TextGrad, {}
    if mode in MRB_VARIANTS:
        algo_cls, variant_kwargs, mode = MRB_VARIANTS[mode]
        extra = {**variant_kwargs, **(mrb_kwargs or {})}
    elif mode in SAIP_VARIANTS:
        algo_cls, extra, mode = TextGradSAIP, dict(SAIP_VARIANTS[mode]), "CCC"
    elif mode in SAI_VARIANTS:
        algo_cls, extra, mode = TextGradSAI, dict(SAI_VARIANTS[mode]), "CCC"
    elif mode in AF_VARIANTS:
        algo_cls, extra, mode = TextGradAF, dict(AF_VARIANTS[mode]), "CCC"
    elif mode in DRO_VARIANTS:
        algo_cls, extra, mode = TextGradDRO, dict(DRO_VARIANTS[mode]), "CCC"
    elif mode in CTG_VARIANTS:
        algo_cls, extra, mode = CTG, dict(CTG_VARIANTS[mode]), "CCC"
    if attr_repeats is not None and issubclass(algo_cls, MRB):
        extra = {**extra, "attribution_repeats": attr_repeats}
    loss_s, grad_s, opt_s = MODE_CONFIGS[mode]
    output_dir = out_root / job_name / "run"
    if (output_dir / f"eval_step_{steps}.parquet").exists():
        print(f"[{job_name}] already complete, skipping")
        return str(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wgc_config.defaults.substep_delay = 0.0
    wgc_config.defaults.batch_invocation_timeout = 6 * 3600.0
    for role in ("task", "optimizer", "gradient", "loss"):
        setattr(wgc_config.defaults, f"{role}_llm_timeout", 1800.0)
        model_cfg = getattr(llm_config, f"{role}_model")
        if model_cfg.reasoning_effort:
            # Reasoning tokens count against max_tokens: use the config's budget, not the 256-token default.
            setattr(wgc_config.defaults, f"{role}_llm_max_tokens", model_cfg.max_tokens)
    if task_max_tokens:
        wgc_config.defaults.task_llm_max_tokens = task_max_tokens  # room for reasoning before the JSON
    if val_metric == "multi":
        extra = {**extra, "objectives": objectives or ["all"]}
    initial_prompt = get_initial_prompt(dataset=dataset, tasks=tasks)
    llms = dict(
        task_llm=create_task_llm(llm_config=llm_config, temperature=0.1),
        optimizer_llm=create_optimizer_llm(llm_config=llm_config, temperature=COMMON_PARAMS["optimizer_llm_temperature"]),
        gradient_llm=create_gradient_llm(llm_config=llm_config, temperature=COMMON_PARAMS["gradient_llm_temperature"]),
        loss_llm=create_loss_llm(llm_config=llm_config, temperature=COMMON_PARAMS["loss_llm_temperature"]),
    )
    if issubclass(algo_cls, MRB):
        # Low-noise counterfactual attribution uses a temperature-0 task LLM.
        llms_extra = {"attribution_llm": create_task_llm(llm_config=llm_config, temperature=0.0)}
        extra = {**extra, **llms_extra}
        llms_to_stop = list(llms.values()) + list(llms_extra.values())
    else:
        llms_to_stop = list(llms.values())
    try:
        algo = algo_cls(
            **llms,
            **extra,
            tasks=tasks,
            steps=steps,
            batch_size=batch_size,
            eval_every=eval_every,
            eval_initial_prompt=True,
            eval_first_step=eval_every == 1,
            eval_last_step=True,
            name=job_name,
            verbosity=1,
            validation_metric=val_metric,
            loss_task_strategy=loss_s,
            gradient_task_strategy=grad_s,
            optimizer_task_strategy=opt_s,
            eval_subset=eval_subset,
            **{**COMMON_PARAMS, "validation_gate_samples": gate_samples},
        )
        algo.train(dataset=dataset, initial_prompt=initial_prompt, output_dir=str(output_dir))
    finally:
        for llm in llms_to_stop:
            try:
                llm.stop()
            except Exception:
                pass
    print(f"[{job_name}] COMPLETED -> {output_dir}")
    return str(output_dir)


def run_single_safe(**kw):
    """Worker-side wrapper: litellm exceptions do not unpickle in the parent, which breaks the whole pool and kills
    every running job. Re-raise as a plain RuntimeError carrying the full worker traceback."""
    try:
        return run_single(**kw)
    except Exception:
        raise RuntimeError(f"[{kw.get('job_name')}] worker traceback:\n{traceback.format_exc()}") from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SummEval")
    ap.add_argument("--modes", nargs="+", default=["SSS", "SSC", "SCC", "CCC", "ST"])
    ap.add_argument("--val", nargs="+", default=["mae"], choices=list(VAL_SHORT))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stagger", type=float, default=5.0, help="seconds between job launches")
    ap.add_argument("--llm-config", default=str(REPO / "expt/configs/llm_local_dsv41.json"))
    ap.add_argument("--out", default=str(REPO / "results/dsv41"))
    ap.add_argument("--tag", default="dsv41")
    ap.add_argument("--rulebook", default=None, help="warm-start rulebook JSON for MRB variants")
    ap.add_argument("--gate-samples", type=int, default=100, help="validation gate rows (also MRB attribution rows)")
    ap.add_argument("--eval-every", type=int, default=1, help="intermediate test evals every N steps")
    ap.add_argument("--eval-subset", type=int, default=None,
                    help="test rows evaluated at intermediate steps (step 0 and the last step use the full test set)")
    ap.add_argument("--attr-repeats", type=int, default=None, help="MRB attribution repeats (default: class default)")
    ap.add_argument("--objectives", nargs="+", default=None,
                    help="objective names or groups for --val multi (default: all); see objectives.build_objectives")
    ap.add_argument("--task-max-tokens", type=int, default=None, help="task LLM max_tokens (reasoning outputs)")
    ap.add_argument("--only", nargs="+", default=None, help="run only these job names")
    ap.add_argument("--mrb-kw", default=None, help="JSON of extra MRB-variant kwargs (e.g. smoke tests)")
    args = ap.parse_args()

    import json

    llm_config = LLMConfig(**json.load(open(args.llm_config)))
    dataset_cls = getattr(ds_module, args.dataset)
    task_names = [t.task_name for t in dataset_cls.tasks]
    out_root = Path(args.out)

    jobs = []
    for seed in args.seeds:
        for val in args.val:
            for mode in args.modes:
                if mode == "ST":
                    # Single-task baseline: CCC on a one-task prompt (as in run_cherrypick.py).
                    for t in task_names:
                        name = f"TextGrad-{args.dataset}-{args.tag}-CCC-val={VAL_SHORT[val]}-task={t}-run_{seed}"
                        jobs.append(dict(job_name=name, mode="CCC", val_metric=val, task_subset=[t]))
                else:
                    name = f"TextGrad-{args.dataset}-{args.tag}-{mode}-val={VAL_SHORT[val]}-run_{seed}"
                    job = dict(job_name=name, mode=mode, val_metric=val, task_subset=None)
                    if mode in MRB_VARIANTS and (args.rulebook or args.mrb_kw):
                        job["mrb_kwargs"] = {**({"rulebook_path": args.rulebook} if args.rulebook else {}),
                                             **json.loads(args.mrb_kw or "{}")}
                    jobs.append(job)

    if args.only:
        jobs = [j for j in jobs if j["job_name"] in set(args.only)]
    print(f"Launching {len(jobs)} jobs with {args.workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {}
        for i, job in enumerate(jobs):
            if 0 < i < args.workers:
                time.sleep(args.stagger)
            fut = ex.submit(
                run_single_safe, llm_config=llm_config, dataset_cls=dataset_cls, steps=args.steps,
                batch_size=args.batch_size, out_root=out_root, gate_samples=args.gate_samples,
                eval_subset=args.eval_subset, attr_repeats=args.attr_repeats, objectives=args.objectives,
                eval_every=args.eval_every,
                task_max_tokens=args.task_max_tokens, **job,
            )
            futs[fut] = job["job_name"]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:
                print(f"[{futs[f]}] FAILED:\n{traceback.format_exc()}")
    print(f"All done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
