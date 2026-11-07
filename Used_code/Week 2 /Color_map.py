# 3. 파리 구역 GeoJSON 데이터 가져오기
geojson_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/arrondissements/exports/geojson?lang=fr&timezone=Europe%2FParis"
response = requests.get(geojson_url)
paris_geo = response.json()

# 1. 매핑용 문자열 컬럼 생성
df['arr_str'] = df['Arrondissement'].astype(str)

# 4. Folium 지도 객체 생성
m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles='CartoDB positron')

# 2. Choropleth 레이어 다시 그리기
folium.Choropleth(
    geo_data=paris_geo,
    name='choropleth',
    data=df,
    columns=['arr_str', 'Cluster'],      # 문자열로 변환한 구역 번호와 클러스터 열
    key_on='feature.properties.c_ar',    # GeoJSON 안의 구역 번호 속성 (보통 c_ar)
    fill_color='YlOrRd',                 
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Cluster (0: Prime, 1: Red Ocean, 2: Target)'
).add_to(m)

# 6. 각 구역의 중심점에 임대료/카페 정보 툴팁(마커) 추가
for i in range(len(df)):
    # GeoJSON에서 해당 구역의 대략적인 중심점 좌표를 추출
    # (주의: 실제 구역의 중심 좌표를 정확히 띄우려면 별도의 위경도 데이터가 필요합니다. 
    pass


folium.GeoJson(
    paris_geo,
    style_function=lambda feature: {
        'fillColor': '#ffffff',
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.1,
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=['l_ar'], 
        aliases=['구역명:'], 
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
    )
).add_to(m)

# 7. 지도 출력 (코랩 화면에 바로 나타남)
m

# 만약 HTML 파일로 저장하고 싶다면 아래 주석을 해제
# m.save('Paris_Target_Clusters.html')
