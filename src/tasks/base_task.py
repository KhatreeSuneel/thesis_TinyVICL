# from dataclasses import dataclass, field
# import pandas as pd
# from typing import Literal

# @dataclass
# class BaseTask:
#     data: pd.DataFrame
#     task_name: str = field(init=True)
#     lossfun: Literal["MSE", "SoftDiceLoss"] = field(init=True)

#     def get_data(self) -> pd.DataFrame:
#         """
#         Override in subclasses: return filtered DataFrame relevant to the task.
#         """
#         return self.data
#     def get_lossfun(self) -> Literal["MSE", "SoftDiceLoss"]:
#         """Simply returns the loss function of the task
#         :return: lossfun: Literal["MSE", "SoftDiceLoss"]
#         """
#         return self.lossfun
#     def get_task_name(self) -> str:
#         """Simply returns the task name
#         :return: task_name: str
#         """
#         return self.task_name

    
   
from dataclasses import dataclass, field
import pandas as pd
from typing import Optional, Tuple, Literal
from abc import ABC, abstractmethod


@dataclass
class BaseTask(ABC):
    data: pd.DataFrame = field(repr=False)
    task_name: str
    lossfun: Literal["MSE", "SoftDiceLoss"]
    shape: Optional[Tuple[int, int]] = field(init=False, default=None)

    def __post_init__(self):
        self.data = self.process_data_for_task()
        self.check_data()
        self.shape = self.data.shape
        self.task_name = self.task_name.upper()
        print(f"[INFO] Initialized task '{self.task_name}' with data shape {self.shape} and loss '{self.lossfun}'")

    @abstractmethod
    def process_data_for_task(self) -> pd.DataFrame:
        """Filter or transform the raw input data for the specific task"""
        raise NotImplementedError

    def check_data(self) -> None:
        """Basic quality check: ensure data is not empty after processing"""
        if self.data.empty:
            raise ValueError(f"Data is empty for task {self.task_name}! Please check filtering logic.")

    def get_data(self) -> pd.DataFrame:
        """Return the processed data"""
        return self.data

    def get_shape(self) -> Tuple[int, int]:
        """Return the shape of the processed data"""
        return self.shape

    def get_task_name(self) -> str:
        """Return the task name"""
        return self.task_name

    def get_lossfun(self) -> Literal["MSE", "SoftDiceLoss"]:
        """Return the configured loss function"""
        return self.lossfun
