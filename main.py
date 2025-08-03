import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from src.utils import load_credit_card_data
from src.em import initialize_parameters, e_step, m_step, compute_log_likelihood

def main():
    """
    # 결과 출력
    print("데이터 로딩 완료")
    print(f"입력 데이터 X shape: {X.shape}")
    print(f"라벨 y 분포: 정상 = {(y==0).sum()}건, 사기{(y==1).sum()}건")
    print(f"이상 비율: {100 * (y==1).sum() / len(y):.4f}%")
    """
    
    """
    # 2. GMM 모델 정의 및 학습
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X)

    # 3. 로그우도 계산 (log p(x_i))
    log_probs = gmm.score_samples(X)

    # 4. threshold 설정 (예: 하위 2%)
    threshold = np.percentile(log_probs, 2)  # 하위 2%를 이상치로 판단
    preds = (log_probs < threshold).astype(int)

    # 5. 성능 평가 (정답 레이블이 있을 경우)
    auc = roc_auc_score(y, preds)
    print(f"ROC_AUC_Score: {auc:.4f}")
    """

    # 1. 데이터 불러오기
    X, y = load_credit_card_data()

    # 2. 하이퍼파라미터 설정

    n_components = 2
    n_iter = 20

    # 3. 파라미터 초기화
    mu, sigma, pi = initialize_parameters(X, n_components)

    # 4. EM 반복
    prev_ll = None
    log_likelihoods = []
    for i in range(n_iter):
        gamma = e_step(X, mu, sigma, pi)
        mu, sigma, pi = m_step(X, gamma)

        log_likelihood = compute_log_likelihood(X, mu, sigma, pi)
        print(f"Iter {i+1}: Log-Likelihood = {log_likelihood:.4f}")

        if prev_ll is not None and abs(log_likelihood - prev_ll) < 1e-4:
            print(f"converged at iter {i+1}")
            break
        prev_ll = log_likelihood
        log_likelihoods.append(log_likelihood)
    
    # 마지막에
    import matplotlib.pyplot as plt
    plt.plot(log_likelihoods)
    plt.title("Log-Likelihood over EM Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()