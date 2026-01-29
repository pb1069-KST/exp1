"""
data/features.py
input_sampled.csv (CIC-IDS2017 기반) 피처 리스트 정의
"""

# 1. 원본 데이터의 피처 순서 그대로 정의 (총 78개)
# Label 컬럼을 제외한 모든 컬럼 이름입니다.
FEATURE_NAMES = [
    "ACK Flag Count",
    "Active Max",
    "Active Mean",
    "Active Min",
    "Active Std",
    "Average Packet Size",
    "Avg Bwd Segment Size",
    "Avg Fwd Segment Size",
    "Bwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Header Length",
    "Bwd IAT Max",
    "Bwd IAT Mean",
    "Bwd IAT Min",
    "Bwd IAT Std",
    "Bwd IAT Total",
    "Bwd PSH Flags",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Min",
    "Bwd Packet Length Std",
    "Bwd Packets/s",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Destination Port",
    "Down/Up Ratio",
    "ECE Flag Count",
    "FIN Flag Count",
    "Flow Bytes/s",
    "Flow Duration",
    "Flow IAT Max",
    "Flow IAT Mean",
    "Flow IAT Min",
    "Flow IAT Std",
    "Flow Packets/s",
    "Fwd Avg Bulk Rate",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Header Length",
    "Fwd Header Length.1",
    "Fwd IAT Max",
    "Fwd IAT Mean",
    "Fwd IAT Min",
    "Fwd IAT Std",
    "Fwd IAT Total",
    "Fwd PSH Flags",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Min",
    "Fwd Packet Length Std",
    "Fwd Packets/s",
    "Fwd URG Flags",
    "Idle Max",
    "Idle Mean",
    "Idle Min",
    "Idle Std",
    "Init_Win_bytes_backward",
    "Init_Win_bytes_forward",
    "Max Packet Length",
    "Min Packet Length",
    "PSH Flag Count",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "RST Flag Count",
    "SYN Flag Count",
    "Subflow Bwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Fwd Packets",
    "Total Backward Packets",
    "Total Fwd Packets",
    "Total Length of Bwd Packets",
    "Total Length of Fwd Packets",
    "URG Flag Count",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
]

# 2. 전처리 시 주의가 필요한 피처 (Infinity/NaN이 발견된 컬럼)
FEATURES_WITH_NULLS = ["Flow Bytes/s", "Flow Packets/s"]

# 3. 데이터 탐색 시 참고할 피처 수
INPUT_DIM = len(FEATURE_NAMES)  # 78

# 4. 클래스 라벨 정보 (LabelEncoder 사용 시 매핑 기준)
LABEL_NAMES = [
    "BENIGN",
    "DDoS",
    "PortScan",
    "Bot",
    "Infiltration",
    "Web Attack Brute Force",
    "Web Attack XSS",
    "Web Attack Sql Injection",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DoS Hulk",
    "DoS GoldenEye",
    "Heartbleed",
]

NUM_CLASSES = len(LABEL_NAMES)  # 15


def get_feature_name(idx):
    """인덱스를 기반으로 피처 이름을 반환 (XAI용)"""
    if idx < len(FEATURE_NAMES):
        return FEATURE_NAMES[idx]
    return f"Unknown_Feature_{idx}"
