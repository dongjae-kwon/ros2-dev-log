import pandas as pd
import numpy as np

# 전처리 및 모델 관련 라이브러리
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 데이터 분할 및 KNN 모델
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 성능 평가
from sklearn.metrics import accuracy_score, classification_report

# 로봇 상태 예측을 위한 예제 데이터 생성
data = {
    "front_distance": [0.5, 1.2, np.nan, 0.3, 2.0, 0.4, 1.8, 0.2, 2.2, 0.6, 1.5, 0.25],
    "motor_temp": [45, 55, 60, 80, np.nan, 85, 50, 90, 48, 78, 58, 88],
    "battery": [90, 80, 70, 40, 85, np.nan, 88, 35, 92, 45, 78, 30],
    "floor_type": ["tile", "carpet", "tile", "wood", np.nan, "wood", "carpet", "tile", "tile", "wood", "carpet", "tile"],
    "status": ["safe", "safe", "safe", "danger", "safe", "danger", "safe", "danger", "safe", "danger", "safe", "danger"]
}

# DataFrame 생성
df = pd.DataFrame(data)

# 입력 데이터(X)와 정답 데이터(y) 분리
X = df[["front_distance", "motor_temp", "battery", "floor_type"]]
y = df["status"]

# 수치형 특성과 범주형 특성 구분
numeric_features = ["front_distance", "motor_temp", "battery"]
categorical_features = ["floor_type"]

# 수치형 데이터 전처리
# 1. 결측값을 중앙값으로 대체
# 2. 표준화 수행
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 범주형 데이터 전처리
# 1. 결측값을 최빈값으로 대체
# 2. 원-핫 인코딩 수행
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 수치형, 범주형 전처리를 하나로 통합
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 전체 머신러닝 파이프라인 구성
# 전처리 후 KNN 분류기 적용
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=3))
])

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,      # 테스트 데이터 25%
    random_state=42,     # 결과 재현 가능
    stratify=y           # 클래스 비율 유지
)

# 모델 학습
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)

# 결과 출력
print("정확도:", accuracy)

print("\n분류 리포트:")
print(classification_report(y_test, y_pred))