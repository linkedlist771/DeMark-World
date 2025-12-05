from typing import List, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler


def find_2d_data_bkps(X: List[Tuple[int, int]]) -> List[int]:
    X_clean = [point if point is not None else (np.nan, np.nan) for point in X]
    X = np.array(X_clean, dtype=float)
    X = pd.DataFrame(X).interpolate("linear").bfill().ffill().to_numpy()
    X_std = StandardScaler().fit_transform(X)
    algo = rpt.KernelCPD(kernel="rbf", jump=1).fit(X_std)
    bkps = algo.predict(pen=10)
    return bkps[:-1]


def get_interval_average_bbox(
    bboxes: List[Tuple[int, int, int, int] | None], bkps: List[int]
) -> List[Tuple[int, int, int, int]]:
    average_bboxes = []
    for left, right in zip(bkps[:-1], bkps[1:]):
        bboxes_interval = bboxes[left:right]
        valid_bboxes = [bbox for bbox in bboxes_interval if bbox is not None]
        if len(valid_bboxes) > 0:
            average_bbox = np.mean(valid_bboxes, axis=0)
            average_bboxes.append(tuple(map(int, average_bbox)))
        else:
            average_bboxes.append(None)
    return average_bboxes


def find_idxs_interval(idxs: List[int], bkps: List[int]) -> List[int]:
    def _find_idx_interval(_idx: int) -> int:
        left = 0
        right = len(bkps) - 2

        while left <= right:
            mid = (left + right) // 2
            if bkps[mid] <= _idx < bkps[mid + 1]:
                return mid
            elif _idx < bkps[mid]:
                right = mid - 1
            else:
                left = mid + 1
        return min(max(left, 0), len(bkps) - 2)

    intervals = []
    for idx in idxs:
        interval_idx = _find_idx_interval(idx)
        intervals.append(interval_idx)
    return intervals



def refine_bkps_by_chunk_size(bkps: List[int], chunk_size: int) -> List[int]:
    result = set()
    for start, end in zip(bkps[:-1], bkps[1:]):
        result.update(range(start, end, chunk_size))
        result.add(end)
    result.add(bkps[0])
    return sorted(result)
