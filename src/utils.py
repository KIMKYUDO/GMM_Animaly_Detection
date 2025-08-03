import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_credit_card_data(path='data/creditcard.csv'):
    """
    CSV 파일을 불러와 전처리하고 X (입력), y (라벨)로 나누어 반환한다.

    전처리:
    - 'Amount' 칼럼 정규화 (평균 0, 표준편차 1)
    - 'Time' 칼럼 제거
    - 'Class' 라벨 분리
    """
    # CSV 파일 로딩
    df = pd.read_csv(path)

    # Amount 칼럼 정규화화
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

    # Time 칼럼 제거
    df = df.drop(columns=['Time'])

    # 입력(X), 라벨(y) 분리
    X = df.drop(columns=['Class']).values
    y = df['Class'].values

    return X, y