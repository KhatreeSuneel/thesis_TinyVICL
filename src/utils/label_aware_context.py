import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms.functional import pil_to_tensor

class ContextPrecomputeManager:
    def __init__(self, save_dir="config/precomputed_contexts", resize_shape=(256,256), context_set_size=7):
        self.save_dir = save_dir
        self.resize_shape = resize_shape
        self.context_set_size = context_set_size
        os.makedirs(save_dir, exist_ok=True)

    def ensure_precomputed(self, task, task_idx):
        if task.get_task_name().lower() != "segmentation":
            return None

        csv_path = os.path.join(self.save_dir, f"task_{task_idx}_context.csv")
        if os.path.exists(csv_path):
            return csv_path

        df = task.get_data()
        # Compute class presence per sample
        class_presence = {}
        for i, row in df.iterrows():
            lbl_path = os.path.join(row["label_dir"], f"{row['label_name']}{row['label_suffix']}")
            lbl = Image.open(lbl_path).resize(self.resize_shape, Resampling.NEAREST)
            class_presence[i] = set(torch.unique(pil_to_tensor(lbl)).tolist())

        ids = list(df.index)
        mappings = []

        for qidx in ids:
            qlabels = class_presence[qidx]
            selected = set()

            # Cover all labels
            for l in qlabels:
                candidates = [idx for idx in ids if idx != qidx and l in class_presence[idx] and idx not in selected]
                if candidates:
                    selected.add(np.random.choice(candidates))

            # Fill remaining to reach context_set_size
            remaining = self.context_set_size - len(selected)
            if remaining > 0:
                remaining_candidates = list(set(ids) - {qidx} - selected)
                if remaining_candidates:
                    extra = np.random.choice(remaining_candidates, size=remaining, replace=False)
                    selected.update(extra)

            # Truncate if somehow more than context_set_size
            selected_list = list(selected)[:self.context_set_size]
            mappings.append({"query_idx": qidx, "context_indices": selected_list})

        pd.DataFrame(mappings).to_csv(csv_path, index=False)
        return csv_path

    def load_csv(self, csv_path):
        return pd.read_csv(csv_path) if csv_path else None
