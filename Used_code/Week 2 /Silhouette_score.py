import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. 1주 차에 완성한 파리 20개 구역 데이터 불러오기
try:
    df = pd.read_csv("Paris_data.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("Paris_data.csv", encoding="euc-kr")

# 사용할 변수명 매핑 (임대료와 카페 개수)
rent_col = '한달 임대료' if '한달 임대료' in df.columns else '제곱미터 당 평균 매매 가격'
cafe_col = 'cafe_count'

# 2. 데이터 전처리 (결측치 제거 및 스케일링)
features = df[[rent_col, cafe_col]].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 3. 최적의 K 찾기 (실루엣 점수 계산)
print("[ 군집 개수(K)별 실루엣 점수 ]")
for k in range(2, 6):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_temp.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    print(f"K={k}: {score:.4f}")

# 4. 최적의 K(3)로 최종 군집화 및 라벨링 해석
print("\n[ 군집별 특성 분석 (K=3) ]")
best_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = best_kmeans.fit_predict(scaled_features)

# 각 군집의 평균 임대료와 카페 개수 확인
cluster_summary = df.groupby('Cluster')[[rent_col, cafe_col]].mean().round(2)
print(cluster_summary)

# 5. 각 구역이 어느 군집에 속하는지 결과 저장 (선택 사항)
df.to_csv("Paris_data_with_clusters.csv", index=False, encoding="utf-8-sig")
print("\n'Paris_data_with_clusters.csv'로 결과가 저장되었습니다.")
