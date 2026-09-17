# 서울시 공공데이터 기반 상권 쇠퇴 징후 감지 및 분석 파이프라인
# 프로젝트명: 도시의 신호를 읽다 - 초밀착 생활권 상권 건강도 지수(BVI) 구축
# 사용 데이터: 2025년 서울시 상권 마스터, 점포-상권, 집객시설-상권 CSV

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# 1. 데이터 로드 및 전처리 (Data Loading & Preprocessing)

def load_and_preprocess_data():    
    
    # 데이터 파일 경로
    path_master = 'C:/python/seoul_commercial_district_master_2025.csv'
    path_stores = 'C:/python/seoul_commercial_district_stores_2025.csv'
    path_attract = 'C:/python/seoul_commercial_district_attraction_facilities_2025.csv'
    
    # 인코딩에 맞게 데이터 로드 (CP949 및 UTF-8-SIG)
    df_master = pd.read_csv(path_master, encoding='cp949')
    df_stores = pd.read_csv(path_stores, encoding='cp949')
    df_attract = pd.read_csv(path_attract, encoding='utf-8-sig')
    
    print(f" - 마스터 상권 수: {df_master['상권_코드'].nunique()}개")
    print(f" - 점포 레코드 수: {len(df_stores):,}건")
    print(f" - 집객시설 레코드 수: {len(df_attract):,}건")
    
    return df_master, df_stores, df_attract


# 2. 피처 엔지니어링 (Feature Engineering & Signal Extraction)
def build_features(df_stores, df_attract, df_master):
    
    # (1) 분기별/상권별 기본 점포 수 집계
    district_q = df_stores.groupby(['상권_코드', '상권_코드_명', '상권_구분_코드_명', '기준_년분기_코드']).agg({
        '전체_점포_수': 'sum',
        '개업_점포_수': 'sum',
        '폐업_점포_수': 'sum',
        '프랜차이즈_점포_수': 'sum',
        '일반_점포_수': 'sum'
    }).reset_index()
    
    # (2) 주요 파생 지표 산출
    eps = 1e-5  # Zero Division 방지
    district_q['폐업률'] = district_q['폐업_점포_수'] / (district_q['전체_점포_수'] + eps)
    district_q['개업률'] = district_q['개업_점포_수'] / (district_q['전체_점포_수'] + eps)
    district_q['프랜차이즈_비율'] = district_q['프랜차이즈_점포_수'] / (district_q['전체_점포_수'] + eps)
    district_q['점포_생존_압력_지수'] = district_q['폐업_점포_수'] / (district_q['개업_점포_수'] + 1.0)
    
    # (3) Shannon Entropy 기반 업종 다양성 지수 계산
    def calc_shannon_entropy(group):
        p = group['전체_점포_수'] / (group['전체_점포_수'].sum() + eps)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    entropy_df = df_stores.groupby(['상권_코드', '기준_년분기_코드']).apply(calc_shannon_entropy).reset_index(name='업종_엔트로피')
    
    # (4) 집객시설 데이터 선택 및 결합
    attract_cols = ['상권_코드', '기준_년분기_코드', '집객시설_수', '관공서_수', '은행_수', '약국_수', '지하철_역_수', '버스_정거장_수']
    df_attract_sub = df_attract[attract_cols].fillna(0)
    
    # (5) 마스터 공간 데이터 선택 (자치구/행정동/좌표)
    master_cols = ['상권_코드', '자치구_코드_명', '행정동_코드_명', '영역_면적', '엑스좌표_값', '와이좌표_값']
    
    # 최종 데이터 조인
    merged = district_q.merge(entropy_df, on=['상권_코드', '기준_년분기_코드'], how='left')
    merged = merged.merge(df_attract_sub, on=['상권_코드', '기준_년분기_코드'], how='left')
    merged = merged.merge(df_master[master_cols], on='상권_코드', how='left')
    
    merged.fillna(0, inplace=True)
    print(f" - 피처 생성 완료: {merged.shape[0]}행 x {merged.shape[1]}열")
    return merged


# 3. 머신러닝 기반 상권 쇠퇴 예측 & BVI (상권 건강도 지수) 구축
def train_decline_model_and_bvi(merged):
    
    # 2025년 1~3분기 선행 지표 평균을 X로 설정
    q1_q3 = merged[merged['기준_년분기_코드'].isin([20251, 20252, 20253])]
    q4 = merged[merged['기준_년분기_코드'] == 20254]
    
    feature_cols = [
        '전체_점포_수', '폐업률', '개업률', '프랜차이즈_비율', 
        '점포_생존_압력_지수', '업종_엔트로피', '집객시설_수', 
        '지하철_역_수', '버스_정거장_수', '영역_면적'
    ]
    
    X_df = q1_q3.groupby('상권_코드')[feature_cols].mean().reset_index()
    
    # 4분기 폐업률 상위 20%를 '쇠퇴 위험 상권(1)'으로 타겟 설정
    q4_threshold = q4['폐업률'].quantile(0.80)
    q4['쇠퇴_위험_라벨'] = (q4['폐업률'] >= q4_threshold).astype(int)
    
    y_df = q4[['상권_코드', '쇠퇴_위험_라벨', '폐업률']].rename(columns={'폐업률': 'Q4_폐업률'})
    train_data = X_df.merge(y_df, on='상권_코드', how='inner')
    
    X = train_data[feature_cols]
    y = train_data['쇠퇴_위험_라벨']
    
    # Random Forest 분류 모델 학습
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    y_pred_prob = rf.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, y_pred_prob)
    print(f" - [모델 성능] Train ROC-AUC Score: {auc_score:.4f}")
    
    # Feature Importance (피처 중요도)
    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\n [피처 중요도 Top 5]:")
    print(importances.head(5).to_string(index=False))
    
    # BVI (Business Vitality Index: 상권 건강도 지수) 계산 (100점 만점 정규화)
    train_data['쇠퇴_위험_확률'] = y_pred_prob
    train_data['상권_건강도_지수(BVI)'] = np.round((1.0 - y_pred_prob) * 100, 2)
    
    # 건강도 등급 부여
    def assign_grade(bvi):
        if bvi >= 70:
            return '안전'
        elif bvi >= 40:
            return '주의'
        else:
            return '위험'
            
    train_data['건강도_등급'] = train_data['상권_건강도_지수(BVI)'].apply(assign_grade)
    
    # 마스터 메타 정보 매핑 (상권명, 자치구, 좌표 등)
    info_cols = ['상권_코드', '상권_코드_명', '상권_구분_코드_명', '자치구_코드_명', '행정동_코드_명', '엑스좌표_값', '와이좌표_값']
    master_info = merged[merged['기준_년분기_코드'] == 20254][info_cols].drop_duplicates('상권_코드')
    
    final_result = train_data.merge(master_info, on='상권_코드', how='left')
    return final_result, importances


# 4. 메인 실행 및 결과 저장
if __name__ == '__main__':
    df_master, df_stores, df_attract = load_and_preprocess_data()
    merged = build_features(df_stores, df_attract, df_master)
    bvi_result, importances = train_decline_model_and_bvi(merged)
    
    # 최종 결과 CSV 저장
    output_filename = 'seoul_commercial_bvi_results_2025.csv'
    bvi_result.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\nBVI 결과가 '{output_filename}'으로 저장되었습니다.")
    
    # 쇠퇴 위험 상권 상위 5개 출력
    print("\n [서울시 쇠퇴 위험 상권 Top 5 (BVI 최하위)]:")
    cols_to_show = ['상권_코드_명', '자치구_코드_명', '상권_구분_코드_명', '상권_건강도_지수(BVI)', '건강도_등급', 'Q4_폐업률']
    top_risk = bvi_result.sort_values(by='상권_건강도_지수(BVI)').head(5)[cols_to_show]
    print(top_risk.to_string(index=False))
