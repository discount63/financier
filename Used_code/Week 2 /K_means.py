import pandas as pd
import folium
import json
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 1주 차에 완성한 파리 20개 구역 데이터 불러오기
df = pd.read_csv("Paris_data.csv", encoding="utf-8")

rent_col = '한달 임대료' if '한달 임대료' in df.columns else '제곱미터 당 평균 매매 가격'
cafe_col = 'cafe_count'

# 2. 전처리 및 K-Means 클러스터링
features = df[[rent_col, cafe_col]].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 최적의 군집 수인 K=3으로 설정 (실루엣점수 근거 하에)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)
