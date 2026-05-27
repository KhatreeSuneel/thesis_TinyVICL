from dataclasses import dataclass
from .base_task import BaseTask
import pandas as pd
from torchvision.transforms import v2
from PIL import Image

@dataclass
class SuperresolutionTask(BaseTask):
    scale_factor: int =4
    def process_data_for_task(self) -> pd.DataFrame:
        df = self.data
        df = df[df["dataset"] == "raise_sr"]
        return df
    def rescale_image(self, image: Image.Image) -> Image.Image:
        """
        Downscale the image by scale_factor and upscale back to original size.
        """
        w, h = image.size

        # Downscale
        lr = image.resize(
            (w // self.scale_factor, h // self.scale_factor),
            resample=Image.BICUBIC
        )

        # Upscale back to original size
        lr_up = lr.resize(
            (w, h),
            resample=Image.BICUBIC
        )

        return lr_up