import os
import torch
import logging
import time
import random
import argparse
import pandas as pd
import numpy as np

import src.data.dataloader as dataloader
import src.tasks.taskloader as taskloader
import src.utils.utils as utils
import src.train.training as training

from src.tasks.base_task import BaseTask
from src.train.metrics import compute_metrics
from src.neuralizer.lightning_model import LightningModel

from scripts.eval_objects import get_eval_objects

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

utils.setup_logging()
logger = logging.getLogger(__name__)

_USE_CPU = False
_DEVICE = torch.device("cpu" if _USE_CPU else "cuda:0")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


# --------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------

def run_eval_pipeline(EVAL_CFG, MODEL_CFGS, device):

    start = time.time()

    # ---------------------------
    # DATA LOADING (MASTER CFG)
    # ---------------------------

    dataloader.load_data(EVAL_CFG)

    metadata_df = dataloader.get_metadata_df(
        EVAL_CFG["dataloader"]["metadata_location"]
    )

    tasks = taskloader.get_tasks(
        df=metadata_df,
        CFG=EVAL_CFG["tasks"],
    )

    test_tasks = taskloader.get_test_tasks(
        df=metadata_df,
        CFG=EVAL_CFG["tasks"],
    )

    # ---------------------------
    # MODELS
    # ---------------------------

    eval_objects = get_eval_objects(
        CFG_1M=MODEL_CFGS["1m"],
        CFG_5M=MODEL_CFGS["5m"],
        device=device,
    )

    df_seen, df_unseen = evaluate_tasks(
        tasks,
        test_tasks,
        eval_objects,
        EVAL_CFG,
        device,
    )

    save_df(df_seen, "seen")
    save_df(df_unseen, "unseen")

    mins, secs = divmod(time.time() - start, 60)
    logger.info(f"Finished in {int(mins)}m {int(secs)}s")


# --------------------------------------------------------
# CORE EVALUATION
# --------------------------------------------------------

def evaluate_tasks(
    tasks,
    test_tasks,
    eval_objects,
    CFG,
    device,
):

    all_tasks = tasks + test_tasks

    seen_rows = []
    unseen_rows = []

    for eval_obj in eval_objects:

        logger.info(f"Evaluating model: {eval_obj.get_name()}")

        model = eval_obj.get_model()
        if model is not None:
            model = model.to(device)

        row_seen = {"model": eval_obj.get_name()}
        row_unseen = {"model": eval_obj.get_name()}

        for task in all_tasks:

            task_name = task.get_task_name()
            suffix = "seen" if task in tasks else "unseen"

            _, _, test_dl = training.get_dataloaders(
                CFG=CFG,
                tasks=[task],
                test_mode=True,
            )

            aggregated = {}

            with torch.no_grad():

                for x, y, ctx_in, ctx_out, lossfun, class_names, task_names in tqdm(test_dl):

                    x, y = x.to(device), y.to(device)
                    ctx_in, ctx_out = ctx_in.to(device), ctx_out.to(device)

                    if model is None:
                        y_pred = random_prediction(ctx_out)
                    else:
                        y_pred = model.forward(x, ctx_in, ctx_out)

                    for b in range(y.size(0)):

                        metrics = compute_metrics(
                            y_true=y[b],
                            y_pred=y_pred[b],
                            ctx_out=ctx_out[b],
                            class_name=class_names[b],
                            device=device,
                        )

                        for k, v in metrics.items():
                            aggregated.setdefault(k, []).append(
                                v.item() if torch.is_tensor(v) else v
                            )

            for k, vals in aggregated.items():
                mean, std = np.mean(vals), np.std(vals)

                if suffix == "seen":
                    row_seen[f"{task_name}_{k}_mean"] = mean
                    row_seen[f"{task_name}_{k}_std"] = std
                else:
                    row_unseen[f"{task_name}_{k}_mean"] = mean
                    row_unseen[f"{task_name}_{k}_std"] = std

        seen_rows.append(row_seen)
        unseen_rows.append(row_unseen)

    return pd.DataFrame(seen_rows), pd.DataFrame(unseen_rows)


# --------------------------------------------------------

def random_prediction(ctx_out):

    idx = torch.randint(
        low=0,
        high=ctx_out.size(1),
        size=(ctx_out.size(0),),
        device=ctx_out.device,
    )

    return ctx_out[
        torch.arange(ctx_out.size(0), device=ctx_out.device),
        idx,
    ]


# --------------------------------------------------------

def save_df(df, flag):

    out_dir = Path("data/evaluation-results/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"results-{flag}.csv"
    df.to_csv(path)
    logger.info(f"Saved: {path}")


# --------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_cfg", default="configs/eval_master.yaml")
    parser.add_argument("--cfg_1m", default="configs/model_1m.yaml")
    parser.add_argument("--cfg_5m", default="configs/model_5m.yaml")
    args = parser.parse_args()

    EVAL_CFG = utils.load_config(args.eval_cfg)

    MODEL_CFGS = {
        "1m": utils.load_config(args.cfg_1m),
        "5m": utils.load_config(args.cfg_5m),
    }

    run_eval_pipeline(EVAL_CFG, MODEL_CFGS, _DEVICE)
