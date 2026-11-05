"""OPRO (Yang et al., 2024) with the official code (third_party/opro-main, unmodified): opt_utils.run_evolution is run as
released with the hyperparameters of the official optimize_instructions.py for BBH-style tasks and a GPT optimizer
(8 instructions per step, temperature 1.0, 3 random exemplars in the meta-prompt, 20 instructions shown, 100 score
buckets, initial instruction "" as for BBH, instruction at the start of the question, instructions over 500 characters
discarded). As in OPRO, the instruction is a short text placed before the task; here the task is the fixed rubric v0,
so the judge sees instruction + v0 + output format.

Our glue: the scorer (eval_utils.evaluate_single_instruction) is replaced by the LLM judge, and an instruction's
training score is its mean per-attribute Spearman on the training items (written to every row's "accuracy", so the
official average equals it); data loading (load_bbh_task_data) returns our items; the optimizer call goes to the same model (llm package).
Changed defaults: num_search_steps 200 -> 12 (budget), old_instruction_score_threshold 0.3 -> 0.0 (Spearman scale);
the file-name helper is made non-ASCII safe (it crashes on characters such as an em dash).
Train items: 160 drawn from fit (OPRO uses a small training split); the final instruction is the one with the best
training score, as in OPRO.
  python baselines/run_opro.py <ds> [--seed 0] [--steps 12]
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "opro-main"))
import gem  # noqa: E402
from opro.optimization import opt_utils  # noqa: E402  (official)
from opro.evaluation import eval_utils  # noqa: E402  (official)

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--ntrain", type=int, default=160); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = "opro"
np.random.seed(a.seed)
OUTD = ds.out(METHOD, a.seed)

train_rows = ds.subset("fit", a.ntrain, seed=a.seed)
raw_data = [dict(input=r["user"], target=json.dumps({x: int(round(v)) for x, v in zip(ds.A, ds.human([r["id"]])[0])}))
            for r in train_rows]
SCORES = {}  # instruction -> training Spearman


def full_instruction(ins):
    return (ins.strip() + "\n\n" if ins.strip() else "") + ds.v0


def evaluate_single_instruction(data, instruction, eval_index_all, **kw):
    idx = list(eval_index_all)
    if not idx:
        return pd.DataFrame({"accuracy": [np.nan]})
    rows = [train_rows[i] for i in idx]
    df = ds.worst_case(ds.judge(full_instruction(instruction), rows, a.workers, strict=False))
    v = np.nanmean(ds.spearman(df))
    SCORES[instruction] = float(v)
    out = pd.DataFrame(dict(raw_answer=df.reason, true_answer=[data[i]["target"] for i in idx]), index=idx)
    out["accuracy"] = max(float(v), 0.0)
    print(f"  score {v:.3f}  ins: {instruction[:100]!r}", flush=True)
    return out


eval_utils.evaluate_single_instruction = evaluate_single_instruction
_to_filename = eval_utils.instruction_to_filename
# the official helper hashes instruction.encode("ascii") and crashes on any non-ASCII character (e.g. an em dash or
# German text); only the file name of the per-instruction results depends on it
eval_utils.instruction_to_filename = lambda ins, md5_hashing=True: _to_filename(
    ins.encode("ascii", "backslashreplace").decode("ascii"), md5_hashing)
opt_utils.eval_utils = eval_utils
eval_utils.load_bbh_task_data = lambda task_name, base_dir=None, qa_format=True: raw_data


def call_optimizer(input_text, temperature=1.0, **kw):
    return [gem.chat([dict(role="user", content=input_text)], temperature=temperature)]


save = f"{OUTD}/opro"
os.makedirs(f"{save}/result_by_instruction", exist_ok=True)
train_index = np.arange(len(raw_data))
opt_utils.run_evolution(
    num_search_steps=a.steps, old_instruction_score_threshold=0.0,
    scorer_llm_dict=dict(model_type="gpt-4", batch_size=1, num_servers=1, max_decode_steps=1024, temperature=0.0,
                         num_decodes=1),
    optimizer_llm_dict=dict(model_type="gpt-4", batch_size=1, num_decodes=1, max_decode_steps=512, temperature=1.0),
    extract_final_answer_by_prompting_again=False, include_qa=False, evaluate_in_parallel=False,
    tasks_all=["judge"], train_ratio=1.0, eval_ratio=0.0, test_ratio=0.0, train_index=train_index,
    eval_index=np.array([], dtype=int), dataset_name="bbh", task_name="judge", num_examples=len(raw_data),
    root_data_folder_path="", optimizer_llm_temperature=1.0, initial_instructions=[""], multiple_choice_tasks=set(),
    raw_data=raw_data, call_scorer_server_func=None, call_optimizer_server_func=call_optimizer, instruction_pos="Q_begin",
    prediction_treat_as_number=False, prediction_treat_as_bool=False,
    result_by_instruction_folder=f"{save}/result_by_instruction", few_shot_qa_pairs=True, num_score_buckets=100,
    max_num_instructions=20, meta_prompt_type="both_instructions_and_exemplars",
    meta_prompt_instructions_before_exemplars=True, few_shot_selection_criteria="random", optimizer_llm_name="gpt-4",
    num_generated_instructions_in_each_step=8, evaluate_generated_ins_on_few_shot=False,
    num_few_shot_questions_for_instruction_refinement=3, evaluate_old_ins_on_few_shot=False, eval_interval=3,
    save_folder=save, verbose=False)

best = max(SCORES, key=SCORES.get)
json.dump(SCORES, open(f"{OUTD}/train_scores.json", "w"), indent=1)
print(f"best training Spearman {SCORES[best]:.3f}: {best!r}")
gem.judge_all_splits(ds, METHOD, a.seed, full_instruction(best), a.workers)
print(a.ds, METHOD, a.seed, "done")
