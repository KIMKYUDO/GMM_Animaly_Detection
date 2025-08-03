# 🧠 GMM 기반 이상 거래 탐지 시스템

이 프로젝트는 **Gaussian Mixture Model (GMM)** 을 활용해 신용카드 거래 데이터에서 **비지도 학습 기반 이상 탐지**를 수행하는 시스템입니다.

## 📌 프로젝트 개요

- **데이터**: Kaggle의 creditcard.csv (PCA 전처리됨)
- **목표**: log-likelihood 기반으로 이상 거래 탐지
- **기법**: GaussianMixture + Thresholding
- **확장**: 직접 EM 구현, ROC 분석, 시각화 등

## 🗂️ 폴더 구조

```bash
GMM_Anomaly_Detection/
├── src/                # 코드
├── data/               # 원본 데이터 (업로드 제외)
├── main.py             # 실행 진입점
├── notebooks/          # 시각화 노트북
├── report/             # 보고서
├── requirements.txt
├── .gitignore
└── README.md