import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from .features import FEATURE_NAMES, LABEL_NAMES, FEATURES_WITH_NULLS


class BSFSDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_and_clean_data(file_path):
    """
    CSV 데이터를 로드하고 NaN/Inf 처리 및 라벨 인코딩을 수행합니다.
    """
    df = pd.read_csv(file_path)

    # 1. 무한대(Inf) 값을 NaN으로 치환 후 처리
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. 결측치(NaN) 처리: 각 컬럼의 평균값으로 채움
    # (네트워크 데이터의 특성상 0으로 채우는 것보다 평균값이 통계적 왜곡이 적음)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 3. 피처(X)와 라벨(y) 분리
    X = df[FEATURE_NAMES].values
    y_str = df["Label"].values

    # 4. 문자열 라벨의 정수 인코딩
    # LABEL_NAMES 순서에 고정하여 인코딩 (BENIGN=0, DDoS=1 ...)
    le = LabelEncoder()
    le.fit(LABEL_NAMES)
    y = le.transform(y_str)

    return X, y, le


def get_dataloader(x, y, batch_size=128, shuffle=True):
    dataset = BSFSDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
