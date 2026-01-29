import numpy as np
from sklearn.preprocessing import StandardScaler


class NetPreProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, x):
        """
        학습 데이터의 평균과 표준편차를 학습하고 변환합니다.
        """
        return self.scaler.fit_transform(x)

    def transform(self, x):
        """
        테스트/검증 데이터를 학습 데이터의 기준에 맞춰 변환합니다.
        """
        return self.scaler.transform(x)

    def inverse_transform(self, x):
        """
        스케일링된 데이터를 원래 수치로 복원합니다 (필요 시).
        """
        return self.scaler.inverse_transform(x)


def preprocess_pipeline(x_train, x_val):
    """
    간편하게 사용할 수 있는 전처리 파이프라인
    """
    processor = NetPreProcessor()
    x_train_scaled = processor.fit_transform(x_train)
    x_val_scaled = processor.transform(x_val)

    return x_train_scaled, x_val_scaled, processor
