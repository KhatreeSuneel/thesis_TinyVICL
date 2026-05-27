from dataclasses import dataclass
from .base_task import BaseTask
import pandas as pd


@dataclass
class EdgeTask(BaseTask):
    def process_data_for_task(self) -> pd.DataFrame:
        df = self.data
        df = df[df["dataset"] == "biped_mdbd_coco"]
        return df