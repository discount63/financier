"""
Week 2 - A: 시뮬레이션 통합 (예지)
1주차 몬테카를로 뼈대 + 깃허브 Paris_data.csv 

- 데이터는 깃허브에서 바로 읽어옴
- 타겟 구: 하린 K-Means Cluster 2 결과 반영 (12, 13, 14, 15, 18, 19, 20구)
- 결과: 구별 BEP 판매량 + 흑자 전환 확률(bep_probability)
"""

import numpy as np
import pandas as pd

DATA_URL = ("https://raw.githubusercontent.com/discount63/financier/main/"
            "csv_files/Paris_data.csv")

# 하린 K-Means 결과 Cluster 2(Target) 구역 (csv_files/Paris_data_with_clusters.csv)
TARGET_ARRONDISSEMENTS = [12, 13, 14, 15, 18, 19, 20]

# ------------------------------------------------------------------
# 1. 가정값 (EUR) - 팀 합의 후 숫자만 수정
# ------------------------------------------------------------------
A = {
    # 잔당
    "price": 2.00,              # 판매가 (2유로 확정)
    "vat_rate": 0.055,          # 프랑스 테이크아웃 부가세 (확인 필요)
    "bean": 0.20, "cup": 0.12, "other": 0.08,
    "card_fee": 0.02,
    # 월 고정비 (임대료 제외)
    "labor": 6000, "utilities": 800, "etc": 500,
    "area_m2": 25,              # 매장 면적
    "days": 30,
    # 수요 가정 (Paris_data에 유동인구가 없어서 인구/카페 수로 추정)
    "coffee_rate": 0.20,        # 인구 1명당 하루 테이크아웃 커피 구매 확률
    "demand_cv_month": 0.20,    # 달마다 수요 수준이 흔들리는 정도
    "demand_cv_day": 0.25,      # 하루하루 판매량 흔들림
}


# ------------------------------------------------------------------
# 2. 데이터 불러오기
# ------------------------------------------------------------------
def load_paris_data(path=None):
    """Paris_data.csv -> 시뮬레이션에 필요한 컬럼만 정리
    같은 폴더에 Paris_data.csv가 있으면 그걸 쓰고, 없으면 깃허브에서 읽음"""
    import os
    if path is None:
        path = "Paris_data.csv" if os.path.exists("Paris_data.csv") else DATA_URL
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.rename(columns={
        "연간 임대료": "rent_m2_year",     # ㎡당 연 임대료 (€)
        "한달 임대료": "rent_m2_month",    # ㎡당 월 임대료 (€)
        "인구": "population",
        "지하철역개수": "metro",
    })


# ------------------------------------------------------------------
# 3. 1주차 함수 (임대료 = ㎡당 월 임대료 × 면적)
# ------------------------------------------------------------------
def contribution_per_cup(a=A):
    """잔당 공헌이익 = 부가세 뺀 판매가 - 변동비"""
    net = a["price"] / (1 + a["vat_rate"])
    var = a["bean"] + a["cup"] + a["other"] + a["price"] * a["card_fee"]
    return net - var


def fixed_cost_monthly(rent_m2_month, a=A):
    """월 고정비 = 월세(㎡당 월 임대료 × 면적) + 인건비 + 공과금 + 기타"""
    return rent_m2_month * a["area_m2"] + a["labor"] + a["utilities"] + a["etc"]


def bep_daily_cups(rent_m2_month, a=A):
    """손익분기 일 판매량 (잔/일)"""
    return fixed_cost_monthly(rent_m2_month, a) / contribution_per_cup(a) / a["days"]


def expected_daily_cups(population, cafe_count, a=A):
    """예상 일 판매량 = 인구 × 구매확률 ÷ (기존 카페 수 + 우리 매장 1)"""
    return population * a["coffee_rate"] / (cafe_count + 1)


# ------------------------------------------------------------------
# 4. 몬테카를로: 10,000번 반복 -> 흑자 전환 확률
# ------------------------------------------------------------------
def simulate_monthly_profit(rent_m2_month, mean_cups, n_sims=10_000, a=A, seed=42):
    rng = np.random.default_rng(seed)
    # 달마다 수요 수준 자체가 다름 (경기, 계절 등)
    month_level = rng.normal(mean_cups, mean_cups * a["demand_cv_month"], size=(n_sims, 1))
    month_level = np.clip(month_level, 0, None)
    # 하루하루 판매량
    daily = rng.normal(month_level, month_level * a["demand_cv_day"],
                       size=(n_sims, a["days"]))
    daily = np.clip(daily, 0, None)
    monthly_cups = daily.sum(axis=1)
    return monthly_cups * contribution_per_cup(a) - fixed_cost_monthly(rent_m2_month, a)


def bep_probability(profit):
    """한 달 수익이 0을 넘은 비율 = 흑자 전환 확률"""
    return (profit > 0).mean()


def run_simulation(df, targets=None, n_sims=10_000, a=A):
    if targets:
        df = df[df["Arrondissement"].isin(targets)]
    rows = []
    for _, r in df.iterrows():
        mean_cups = expected_daily_cups(r["population"], r["cafe_count"], a)
        profit = simulate_monthly_profit(r["rent_m2_month"], mean_cups, n_sims, a)
        rows.append({
            "구": int(r["Arrondissement"]),
            "월세(€)": r["rent_m2_month"] * a["area_m2"],
            "BEP 판매량(잔/일)": bep_daily_cups(r["rent_m2_month"], a),
            "예상 판매량(잔/일)": mean_cups,
            "평균 월 순이익(€)": profit.mean(),
            "하위 5%(€)": np.percentile(profit, 5),
            "상위 5%(€)": np.percentile(profit, 95),
            "bep_probability": bep_probability(profit),
        })
    return pd.DataFrame(rows).sort_values("bep_probability", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:,.2f}".format)

    df = load_paris_data()
    print(f"잔당 공헌이익: {contribution_per_cup():.2f}€\n")

    print("=== 타겟 상권 ===")
    print(run_simulation(df, TARGET_ARRONDISSEMENTS).to_string(index=False))

    print("\n=== 참고: 파리 20개 구 전체 ===")
    print(run_simulation(df).to_string(index=False))
