import streamlit as st

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

st.header("코로나 데이터 분석")
st.write("---")


st.write("### 프로젝트 목표") 
st.write("* 서울시 코로나 19 확진자 현황 데이터를 분석하여 유의미한 정보 도출")
st.write("* 탐색적 데이터 분석을 수행하기 위한 데이터 정제, 특성 엔지니어링, 시각화 방법 학습")
st.write("---")


st.write("### 프로젝트 목표")
st.write("* https://www.data.go.kr/tcs/dss/selectFileDataDetailView.do?publicDataPk=15063273")
st.write("---")


st.write("#### 프로젝트 개요")
text = """
2020년 초에 발생한 코로나19 바이러스는 세계적으로 대유행하였고 이에 대한 많은 분석이 이루어지고 있습니다. 유행 초기엔 이를 분석할 데이터가 충분하지 않았지만 최근에는 다양한 데이터 기관에서 코로나 관련 데이터를 공공으로 제공하고 있습니다. 
이번 프로젝트에서는 국내 공공데이터 포털에서 제공하는 `서울시 코로나19 확진자 현황` 데이터를 바탕으로 탐색적 데이터 분석을 수행해보겠습니다. 국내 데이터 중 확진자 비율이 제일 높고 사람이 제일 많은 서울시의 데이터를 선정하였으며, 이를 바탕으로 코로나19의 확진 추이 및 환자 특성에 대해서 데이터를 바탕으로 알아봅시다.
"""
st.write(text)


image = Image.open('./img/coronavirus.jpg')
st.image(image, caption='출처 : pixabay')


st.write("---")


st.write("### 1. 데이터 읽기")
st.write("`pandas` 라이브러리를 사용하여 데이터를 읽고 어떠한 데이터가 저장되어 있는지 확인합니다.")

# 데이터 프레임 출력
corona_all = pd.read_csv('./data/서울시 코로나19 확진자 현황.csv')
st.dataframe(corona_all)


st.write("데이터프레임 크기 : ", corona_all.shape)
st.write("---")


st.write("### 2. 데이터 정제")
text = """
데이터를 읽고 확인했다면 **결측값(missing value)**, **이상치(outlier)** 를 처리하는 데이터 정제 과정을 수행해봅시다.
"""
st.write(text)


text = """
corona_all 데이터프레임의 결측값을 확인해보겠습니다
"""
st.write(text)
st.table(corona_all.isnull().sum())


text = """
이 중에서 데이터가 존재하지 않는 `국적`, `환자정보`, `조치사항` 의 colums을 삭제합니다
"""
st.write(text)


corona_del_col = corona_all.drop(columns=['국적', '환자정보', '조치사항'])


text = """
`국적`, `환자정보`, `조치사항` column 삭제 결과
"""
st.write(text)

# 데이터프레임 출력
st.dataframe(corona_del_col)
st.write("데이터프레임 크기 : ", corona_del_col.shape)

text = """
`국적`, `환자정보`, `조치사항`column 삭제 결과(결측치 확인)
"""
st.write(text)

# 테이블 출력
st.table(corona_del_col.isnull().sum())
st.write("---")


st.write("### 3. 데이터 시각화")
text = """
결측값을 처리한 `corona_del_col` 데이터를 바탕으로 각 column의 변수별로 어떠한 데이터 분포를 하고 있는지 시각화를 통하여 알아봅시다.
"""
st.write(text)


st.write("#### 3.1. 확진일")
text = """
`확진일` 데이터를 간단히 출력해보면 `월.일` 형태의 날짜 형식임을 알 수 있습니다.
월별, 일별 분석을 위해서는 문자열 형식의 데이터를 나누어 숫자 형 데이터로 변환해 보겠습니다.
"""
st.write(text)


# 테이블 출력
st.table(corona_del_col['확진일'].head())


st.write("##### `확진일` 데이터를 `month`, `day` 데이터로 나누기")
text = """
`확진일`에 저장된 문자열 데이터를 나누어 `month`, `day` column에 int64 형태로 저장해 봅시다.
"""
st.write(text)

month = []
day = []

for data in corona_del_col['확진일']:
    m, d = data.split('.')[:2]
    
    month.append(int(m))
    day.append(int(d))

corona_del_col['month'] = month
corona_del_col['day'] = day

text = """
`month`, `day` 컬럼이 추가된 것을 확인할 수 있습니다.
"""
st.write(text)

# 데이터프레임 출력
st.dataframe(corona_del_col)


st.write("##### 월별 확진자 수 출력")
text = """
나누어진 `month`의 데이터를 바탕으로 달별 확진자 수를 막대그래프로 출력해 보겠습니다.
"""
st.write(text)


# 그래프 출력
st.bar_chart(corona_del_col['month'].value_counts())


st.write("##### 월별 확진자 수 출력")
text = """
나누어진 `month`의 데이터를 바탕으로 달별 확진자 수를 막대그래프로 출력해 보겠습니다.
"""
st.write(text)

order_day = sorted(corona_del_col['day'].unique())
august = corona_del_col[corona_del_col['month'] == 8]

# 그래프 출력
st.bar_chart(august['day'].value_counts())


st.write("#### 3.2. 지역")
text = """
`지역` 데이터를 간단히 출력해보면 `oo구` 형태의 문자열 데이터임을 알 수 있습니다.
무작위 10개의 데이터를 뽑아서 확인해보겠습니다.
"""
st.write(text)

# 테이블 출력
st.table(corona_del_col['지역'].sample(10))

st.write("##### 지역별 확진자 수 출력")
text = """
이번에는 지역별로 확진자가 얼마나 있는지 막대그래프로 출력해 봅시다.
"""
st.write(text)


# 그래프 출력
st.bar_chart(corona_del_col['지역'].value_counts())


st.write("##### 지역 이상치 데이터 처리")
text = """
출력된 그래프를 보면 `종랑구`라는 잘못된 데이터와 `한국`이라는 지역과는 맞지 않는 데이터가 있음을 알 수 있습니다.
기존 지역 데이터 특성에 맞도록 데이터를 수정하겠습니다.
* 종랑구 → 중랑구
* 한국 → 기타
"""
st.write(text)

corona_out_region = corona_del_col.replace({'종랑구': '중랑구', '한국': '기타'})

# 데이터프레임 출력
st.dataframe(corona_out_region)

# 그래프 출력
st.bar_chart(corona_out_region['지역'].value_counts())


st.write("##### 8월달 지역별 확진자 수 출력")
text = """
감염자가 많았던 8월에는 지역별로 확진자가 어떻게 분포되어 있는지 막대그래프로 출력해 봅시다.
"""
st.write(text)

# 데이터프레임 출력
august_region = corona_out_region[corona_out_region['month'] == 8]
st.dataframe(august_region)

# 그래프 출력
st.bar_chart(august_region['지역'].value_counts())


st.write("##### 월별 관악구 확진자 수 출력")
text = """
이번에는 확진자가 가장 많았던 `관악구` 내의 확진자 수가 월별로 어떻게 증가했는지 그 분포를 막대그래프로 출력해 봅시다.
"""
st.write(text)

st.bar_chart(corona_out_region[corona_out_region['지역'] == '관악구']['month'].value_counts())


import folium
from streamlit_folium import st_folium

CRS = pd.read_csv("./data/서울시 행정구역 시군구 정보 (좌표계_ WGS1984).csv")

corona_seoul = corona_out_region.drop(corona_out_region[corona_out_region['지역'] == '타시도'].index)
corona_seoul = corona_seoul.drop(corona_out_region[corona_out_region['지역'] == '기타'].index)

# 서울 가운데 좌표를 잡아 지도를 출력합니다.
map_osm = folium.Map(location=[37.529622, 126.984307], zoom_start=11)

# 지역 정보를 set 함수를 사용하여 25개 고유의 지역을 뽑아냅니다.
for region in set(corona_seoul['지역']):
    # 해당 지역의 데이터 개수를 count에 저장합니다.
    count = len(corona_seoul[corona_seoul['지역'] == region])

    # 해당 지역의 데이터를 CRS에서 뽑아냅니다.
    CRS_region = CRS[CRS['시군구명_한글'] == region]

    # CircleMarker를 사용하여 지역마다 원형마커를 생성합니다.
    marker = folium.CircleMarker(
        [CRS_region['위도'], CRS_region['경도']],   # 위치
        radius=(count / 10) + 10,       # 범위
        color='#3186cc',            # 선 색상
        fill_color='#3186cc',       # 면 색상
        popup=' '.join((region, str(count), '명'))
    ) # 팝업 설정
    
    # 생성한 원형마커를 지도에 추가합니다.
    marker.add_to(map_osm)

st.write("##### 월별 관악구 확진자 수 출력")
text = """
`folium` 라이브러리와 folium를 streamlit으로 표현할 수 있는 `streamlit_folium` 라이브러리를 이용해서 지도에 확진자를 원형 마커로 출력해보겠습니다
"""
st.write(text)

# 지도 출력
st_data = st_folium(map_osm, width = 1000)


st.write("#### 3.3. 여행력")

text = """
`여행력` 데이터를 간단히 출력해보면 `NaN`과 해외 지역명의 문자열 데이터로 구성된 것을 알 수 있습니다.
"""
st.write(text)

# 데이터프레임 출력
st.dataframe(corona_out_region['여행력'])

st.write("##### 여행력 있다 vs 없다 출력")
text = """
먼저 여행력이 있는 사람과 없는 사람이 어느 정도인지를 비교해 보겠습니다. 
여행력이 없는 사람은 `NaN`에 해당되는 사람으로 비어있는 데이터의 수를 세어 계산합니다. 
"""
st.write(text)

sum_travel_no = sum(corona_out_region['여행력'].isnull())
st.write(f"- **여행력이 없는 사람의 수:** {sum_travel_no}")

# 전체 샘플 수를 구합니다.
sum_travel_all = len(corona_out_region['여행력'])
st.write(f"- **전체 확진자 수:** {sum_travel_all}")

# 여행력이 있는 사람들의 수를 계산합니다.
sum_travel_yes = sum_travel_all - sum_travel_no
st.write(f"- **여행력이 있는 사람의 수:** {sum_travel_yes}")

sum_travel_df = pd.DataFrame(
    {
     "있다" : sum_travel_yes, 
     "없다" : sum_travel_no
    },
    index= ["여행여부"]
)

# 그래프 출력
st.bar_chart(sum_travel_df)


st.write("##### 여행력 분포 출력")
text = """
이번에는 어떠한 여행지들이 있는 그 분포를 출력해 봅시다.
먼저 459개 여행지 데이터에서 중복을 제외한 모든 종료의 여행지를 출력해 봅시다.
"""
st.write(text)

# 데이터프레임 출력
st.dataframe(corona_out_region['여행력'].unique())


st.write("##### 여행력 분포 출력")
text = """
여행지 데이터 중에 `21263`이라는 알 수 없는 여행지의 정보가 있기에 그 데이터를 `기타`로 변경하고 그래프로 표현해보겠습니다.
"""
st.write(text)


# 그래프 출력
st.bar_chart(corona_out_region.replace({'21263': '기타'})['여행력'].value_counts())


st.write("#### 3.4. 접촉력")
st.write("##### 여행력 분포 출력")
text = """
`접촉력` 데이터를 출력해보면 코로나를 접촉한 방식을 설명하는 문자열 데이터임을 알 수 있습니다.
"""
st.write(text)

# 테이블 출력
st.table(corona_out_region['접촉력'].sample(10))


st.write("##### 접촉력 도수분포표")
text = """
확진자의 접촉력은 다양하게 기록되었습니다. 그래프를 사용하여 정리해 봅시다.
"""
st.write(text)

# 그래프 출력
st.bar_chart(corona_out_region['접촉력'].value_counts())

text = """
슬라이드를 움직이면 접촉력이 높은 순서대로의 순위를 볼 수 있습니다
"""
st.write(text)

count = st.slider('보고 싶은 차트의 개수를 선택해주세요.(최대 30개)', 1, 30, 10)


# 그래프 출력
st.bar_chart(corona_out_region['접촉력'].value_counts()[:count])


st.write("#### 3.5. 상태")
st.write("##### 상태별 확진자 수 출력")
text = """
데이터가 어떻게 이루어져 있는지 정확히 알기 위해, 중복을 제외한 모든 종류의 상태와 결과를 출력해 봅시다.
"""
st.write(text)

sum_status_death = len(corona_out_region[corona_out_region['상태'] == '사망'])
st.write(f"- **사망자 수:** {sum_status_death}")

sum_status_discharge = len(corona_out_region[corona_out_region['상태'] == '퇴원'])
st.write(f"- **퇴원자 수:** {sum_status_discharge}")

sum_status_nan = sum(corona_out_region['상태'].isnull())
st.write(f"- **상태를 알 수 없는 사람들의 수 :** {sum_status_nan}")


text = """
최종적으로 상태에 따른 확진자 수를 막대 그래프로 표현해 봅시다.
"""
st.write(text)


sum_status_df = pd.DataFrame(
    {
     "퇴원" : sum_status_discharge, 
     "사망" : sum_status_death,
     "알 수 없음" : sum_status_nan
    },
    index= ["상태여부"]
)

options = st.multiselect(
    '보고 싶은 데이터를 선택하세요(체크된 사항이 없으면 "퇴원"데이터만 보여집니다.)',
    ['퇴원', '사망', '알 수 없음'],
    ['퇴원']
)

if len(options) == 0 :
    options = ["퇴원"]
else :
    pass

st.bar_chart(sum_status_df[options])