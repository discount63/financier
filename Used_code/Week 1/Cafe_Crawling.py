pip install osmnx pandas

import osmnx as ox
import pandas as pd
import time

# 1. 파리의 20개 구(Arrondissement) 이름 리스트 생성
arrondissements = [
    f"Paris {i}er Arrondissement, France" if i == 1 
    else f"Paris {i}e Arrondissement, France" 
    for i in range(1, 21)
]

# 결과를 저장할 빈 리스트
cafe_counts = []

print("파리 구역별 카페 개수 크롤링을 시작합니다...")

# 2. 각 구역을 순회하며 카페 개수 카운트
for place in arrondissements:
    print(f"[{place}] 검색 중...")
    try:
        # 해당 구역의 폴리곤 내에서 'amenity'가 'cafe'인 시설 데이터 가져오기
        tags = {'amenity': 'cafe'}
        # OSM 데이터를 가져옵니다. (geometry 기반)
        gdf = ox.features_from_place(place, tags)
        
        # 총 카페 개수 계산
        count = len(gdf)
        
    except Exception as e:
        # 해당 구역에 카페가 검색되지 않거나 에러가 발생한 경우 0으로 처리
        print(f" - {place} 검색 실패 또는 결과 없음: {e}")
        count = 0
        
    cafe_counts.append({'libelle_geo': place, 'cafe_count': count})
    
    # 서버에 부담을 주지 않기 위해 짧은 대기 시간 추가
    time.sleep(1)

# 3. 데이터프레임으로 변환 및 결과 출력
df_cafe = pd.DataFrame(cafe_counts)
print("\n--- 크롤링 완료! ---")
print(df_cafe.head(20))

# 4. CSV 파일로 저장 (팀원 A의 데이터와 병합하기 위함)
df_cafe.to_csv("Paris_cafe_counts.csv", index=False, encoding='utf-8-sig')
print("Paris_cafe_counts.csv 파일로 저장되었습니다.")
