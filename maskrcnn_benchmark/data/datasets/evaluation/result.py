from enum import IntEnum


class EvalType(IntEnum):
    TRAIN = 1
    VALID = 2
    TEST = 3

class Results:
    pass

class Result:
    def __init__(self, eval_type: EvalType, epoch: int) -> None:
        self.eval_type = eval_type
        self.epoch = epoch


class VGResult(Result):
    def __init__(self, eval_type: EvalType, epoch: int,
                 mean_recall_20: float, mean_recall_50: float, mean_recall_100: float, 
                 recall_20: float, recall_50: float, recall_100: float,
                 ngc_recall_20: float, ngc_recall_50: float, ngc_recall_100: float,
                 zero_shot_recall_20: float, zero_shot_recall_50: float, zero_shot_recall_100: float,
                 ngc_zero_shot_recall_20: float, ngc_zero_shot_recall_50: float, ngc_zero_shot_recall_100: float,
                ) -> None:
        super().__init__(eval_type, epoch)
        self.data = {
            'mean_recall_20': mean_recall_20,
            'mean_recall_50': mean_recall_50,
            'mean_recall_100': mean_recall_100,
            'recall_20': recall_20,
            'recall_50': recall_50,
            'recall_100': recall_100,
            'ngc_recall_20': ngc_recall_20,
            'ngc_recall_50': ngc_recall_50,
            'ngc_recall_100': ngc_recall_100,
            'zero_shot_recall_20': zero_shot_recall_20,
            'zero_shot_recall_50': zero_shot_recall_50,
            'zero_shot_recall_100': zero_shot_recall_100,
            'ngc_zero_shot_recall_20': ngc_zero_shot_recall_20,
            'ngc_zero_shot_recall_50': ngc_zero_shot_recall_50,
            'ngc_zero_shot_recall_100': ngc_zero_shot_recall_100,
        }

