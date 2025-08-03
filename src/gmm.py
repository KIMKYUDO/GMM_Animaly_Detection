import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

def train_and_detect(X, y, n_components=4, contamination=0.001):
    """
    GMM을 학습하고 이상치 탐지 수행
    - X: 입력 feature
    - y: 실제 레이블 (0=정상, 1=이상치)
    - n_components: GMM의 클러스터 개수
    - contamination: 이상치 비율에 따른 threshold 설정
    """
    # GMM 모델 학습
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)

    # 각 샘플의 로그우도 계산
    log_probs = gmm.score_samples(X)

    # 로그우도가 가장 낮은 상위 contamination%를 이상치로 간주
    threshold = np.percentile(log_probs, 100 * contamination)
    y_pred = (log_probs < threshold).astype(int)

    # 성능 출력 (정답 레이블 y가 있을 때만)
    if y is not None:
        auc = roc_auc_score(y, y_pred)
        print(f"AUC: {auc:.4f}")
        print(f"Detected outliers: {np.sum(y_pred)} / {len(y_pred)}")
    else:
        print("레이블이 없으므로 성능 평가 생략")

    return y_pred, log_probs, threshold