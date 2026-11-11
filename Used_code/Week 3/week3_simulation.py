"""
Week 3 - A: 타겟 구역 실제 임대료 + 정규분포 판매량 몬테카를로 (예지)

데이터: csv_files/Paris_data_with_clusters.csv (하린 K-Means 결과)
  Cluster 0 = Prime(비싼 핵심 상권), 1 = Red Ocean, 2 = Target(블루오션)

액션
 1) 2주차에 고른 타겟 구역(Cluster 2)의 실제 임대료 -> 고정비로 세팅
 2) 하루 판매량 ~ 정규분포 (타겟 평균 400잔, 표준편차 = 평균의 20%)
    구별 평균은 인구 ÷ (카페 수 + 1) 비율로 나눔
 3) numpy 난수로 하루 영업 상황 10,000번 시뮬레이션
"""

import os
import numpy as np
import pandas as pd

DATA_URL = ("https://raw.githubusercontent.com/discount63/financier/main/"
            "csv_files/Paris_data_with_clusters.csv")
TARGET_CLUSTER = 2

# ------------------------------------------------------------------
# 가정값 (EUR) - 팀 합의 후 숫자만 수정
# ------------------------------------------------------------------
A = {
    # 판매량 분포
    "cups_mean": 400,           # 하루 평균 판매량 (잔)
    "cups_std": 80,             # 표준편차 (잔)
    "n_sims": 10_000,           # 시뮬레이션 횟수
    # 잔당
    "price": 2.00,              # 판매가 (2유로 확정)
    "vat_rate": 0.10,           # 부가세 10% (컵 커피는 테이크아웃도 10%, Zayd 확인)
    "bean": 0.20, "cup": 0.12, "other": 0.08,
    "card_fee": 0.02,
    # 월 고정비 (임대료 제외)
    "labor": 6000,              # 인건비 (최저임금 정규직 약 3명, 팀 결정으로 유지)
    "utilities": 800, "etc": 500,
    "area_m2": 25,              # 매장 면적 (테이크아웃 카운터 기준 적당, Zayd 확인)
    # 상가 임대료 보정: 파리 동네 상가 ㎡당 연 400~900€ (Zayd, 파리 변호사회 2026 자료)
    "retail_rent_min": 400, "retail_rent_max": 900,
    "days": 30,                 # 월 영업일
}


def load_data(path=None):
    """같은 폴더에 CSV가 있으면 그걸, 없으면 깃허브에서 읽음"""
    if path is None:
        local = "Paris_data_with_clusters.csv"
        path = local if os.path.exists(local) else DATA_URL
    df = pd.read_csv(path, encoding="utf-8-sig")
    return add_retail_rent(df)


def add_retail_rent(df, a=A):
    """CSV 임대료(매매가 × 4% 추정, 주거용 기준)는 상가 임대료보다 낮음.
    20개 구의 '연간 임대료' 순위는 그대로 두고, 범위만 상가 시세(㎡당 연 400~900€)로 옮김.
    -> 'rent_m2_month' = 보정된 ㎡당 월 임대료"""
    r = df["연간 임대료"]
    scaled = a["retail_rent_min"] + (r - r.min()) / (r.max() - r.min()) * (
        a["retail_rent_max"] - a["retail_rent_min"])
    df["rent_m2_year_retail"] = scaled
    df["rent_m2_month"] = scaled / 12
    return df


def contribution_per_cup(a=A):
    """잔당 공헌이익 = 부가세 뺀 판매가 - 변동비"""
    net = a["price"] / (1 + a["vat_rate"])
    var = a["bean"] + a["cup"] + a["other"] + a["price"] * a["card_fee"]
    return net - var


def monthly_rent(rent_m2_month, a=A):
    """실제 임대료(㎡당 월 임대료) × 매장 면적 = 월세"""
    return rent_m2_month * a["area_m2"]


def fixed_cost_daily(rent_m2_month, a=A):
    """하루 고정비 = (월세 + 인건비 + 공과금 + 기타) / 영업일"""
    monthly = monthly_rent(rent_m2_month, a) + a["labor"] + a["utilities"] + a["etc"]
    return monthly / a["days"]


def simulate_daily_profit(rent_m2_month, a=A, seed=42):
    """하루 영업 10,000번: 판매량 ~ N(400, 80) -> 하루 순이익 배열"""
    rng = np.random.default_rng(seed)
    cups = rng.normal(a["cups_mean"], a["cups_std"], size=a["n_sims"])
    cups = np.clip(cups, 0, None)                      # 음수 판매량 방지
    return cups * contribution_per_cup(a) - fixed_cost_daily(rent_m2_month, a)


def district_cups_mean(target, a=A):
    """구별 평균 판매량: 인구 ÷ (카페 수 + 1) 비율대로 나누되,
    타겟 구 전체 평균이 cups_mean(400잔)이 되도록 맞춤"""
    raw = target["인구"] / (target["cafe_count"] + 1)
    return raw / raw.mean() * a["cups_mean"]


def run(df, a=A):
    """구별 평균 판매량 = 인구·카페 수 기반 (타겟 평균 400잔), 표준편차 = 평균의 20%(80/400)"""
    target = df[df["Cluster"] == TARGET_CLUSTER].copy()
    target["cups_mean"] = district_cups_mean(target, a)
    rows = []
    for _, r in target.iterrows():
        a_r = dict(a, cups_mean=r["cups_mean"],
                   cups_std=r["cups_mean"] * a["cups_std"] / a["cups_mean"])
        profit = simulate_daily_profit(r["rent_m2_month"], a_r)
        fixed = fixed_cost_daily(r["rent_m2_month"], a)
        rows.append({
            "구": int(r["Arrondissement"]),
            "월세(€)": monthly_rent(r["rent_m2_month"], a),
            "평균 판매량(잔/일)": r["cups_mean"],
            "하루 고정비(€)": fixed,
            "BEP(잔/일)": fixed / contribution_per_cup(a),
            "하루 평균 순이익(€)": profit.mean(),
            "하위 5%(€)": np.percentile(profit, 5),
            "상위 5%(€)": np.percentile(profit, 95),
            "흑자 확률": (profit > 0).mean(),
            "월 환산 순이익(€)": profit.mean() * a["days"],
        })
    return pd.DataFrame(rows).sort_values("하루 평균 순이익(€)", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:,.2f}".format)

    df = load_data()
    print(f"타겟 구역(Cluster {TARGET_CLUSTER}): "
          f"{sorted(df.loc[df['Cluster'] == TARGET_CLUSTER, 'Arrondissement'].tolist())}")
    print(f"{A['n_sims']:,}회 시뮬레이션 | "
          f"잔당 공헌이익 {contribution_per_cup():.2f}€\n")
    print("구별 평균 판매량: 인구 ÷ (카페 수 + 1) 비율, 타겟 평균 400잔, 표준편차 = 평균의 20%\n")
    print(run(df).to_string(index=False))
