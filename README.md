# 🛠️ 반도체 공정 센서 데이터 기반 Defect 원인 추적 및 수율(Yield) 예측 시스템

> **Target JD**: 삼성전자 DS부문 메모리사업부 반도체공정기술  
> **JD 키워드 매핑**: 빅데이터 분석, Defect 개선 Engineering, 공정/설비 자동화, 수율 예측, Python 통계 Tool

---

## 1. Project Overview

반도체 Fab 라인의 DMI(Defect-Metrology-Inspection) 시스템에서 수집된 **590개 채널의 Metrology 센서 데이터**를 분석하여, 수율(Yield) 저하를 유발하는 핵심 Defect 변수를 발굴하고 불량 예측 모델을 구현한 실무형 데이터 분석 프로젝트.

단순 ML 실습이 아닌, **공정 엔지니어 관점**에서 "어떤 센서의 이상 거동이 불량을 유발하는가"를 데이터로 역추적(Traceback)하는 것을 목표로 한다.

- **Dataset**: [UCI SECOM Semiconductor Manufacturing Dataset](https://archive.ics.uci.edu/dataset/179/secom)
- **Samples**: 1,567개 웨이퍼 × 590개 센서 피처
- **Class Distribution**: 정상(-1) 1,463개 (93.4%) / 불량(1) 104개 (6.6%)

---

## 2. 반도체 데이터 3대 난제 & 해결 전략

실제 Fab 데이터가 가진 고유한 3가지 난제를 식별하고 엔지니어링 관점에서 해결했다.

### 난제 1. 극단적 클래스 불균형 (Class Imbalance)

| 구분 | 내용 |
|------|------|
| **문제** | 불량률 6.6% → 모델이 전부 정상으로 예측해도 Accuracy 93% 달성 (통계적 함정) |
| **해결** | SMOTE 오버샘플링으로 클래스 균형 강제 조정 + Accuracy 대신 **Recall / F1 / AUC-PR** 평가지표 채택 |
| **이유** | 불량을 정상으로 오판(FN) 시 후공정(Packaging) 진행 → 고객사 클레임 발생. 오판 비용 최소화가 목표 |

### 난제 2. 대량 결측치 (Missing Values)

| 구분 | 내용 |
|------|------|
| **문제** | 일부 센서 결측치 91% 초과 — 설비 센서 오작동, 측정 주기 불일치 등 Fab 환경의 전형적 특성 |
| **해결** | 결측치 40% 초과 센서 32개 선제 제거 → 나머지는 **KNN Imputer** 적용 (공정 파라미터 간 물리적 개연성 유지) |
| **이유** | 단순 평균값 대체는 센서 간 물리적 연관성 파괴. 공정 데이터는 인접 파라미터 간 상관성이 높아 KNN이 적합 |

### 난제 3. 590개 고차원 노이즈 (High-Dimensional Noise)

| 구분 | 내용 |
|------|------|
| **문제** | 590개 센서 중 고정값 센서, 불량과 무관한 노이즈 센서 대량 포함 |
| **해결** | **Variance Threshold** (분산 0.01 이하 제거) → **XGBoost Feature Importance**로 핵심 센서 30개 추출 |
| **이유** | 불필요한 피처는 모델 성능 저하 + 과적합 유발. 도메인 지식 기반 2단계 필터링으로 신호/노이즈 분리 |

---

## 3. 분석 파이프라인

```
Raw Data (590 sensors)
    │
    ▼
[EDA] 클래스 불균형 확인 / 결측치 분포 분석
    │
    ▼
[전처리] 결측치 40% 초과 제거 (590 → 558개) → KNN Imputer
    │
    ▼
[Feature Selection] Variance Threshold → XGBoost Feature Importance → Top 30 센서
    │
    ▼
[SMOTE] 클래스 불균형 해소 (정상:불량 = 1:1)
    │
    ▼
[모델링] XGBoost + GridSearchCV (Recall 최적화)
    │
    ▼
[평가] Confusion Matrix / Recall / F1 / AUC-PR
    │
    ▼
[Defect 역추적] Feature Importance 기반 핵심 센서 이상 거동 시각화
```

---

## 4. 실험 결과 비교

| 실험 | 알고리즘 | 전처리 기법 | Recall | F1 | 비고 |
|------|---------|-----------|--------|-----|------|
| 01 | XGBoost Baseline | 결측치 중앙값 대체 | 0.12 | 0.21 | 불균형 미처리 → 불량 검출 실패 |
| 02 | XGBoost + SMOTE | 결측치 제거 + SMOTE | 0.67 | 0.22 | 불량 검출 개선, Precision 낮음 |
| 03 | **XGBoost + SMOTE + Feature Selection** | **KNN Imputer + Variance Threshold + Top 30** | **0.67** | **0.22** | **Recall 최적화 모델 (scale_pos_weight=10)** |

> **핵심 인사이트**: CV Recall 0.994 vs Test Recall 0.67 → SMOTE 기반 과적합(Overfitting) 확인.  
> 개선 방향: LightGBM + Threshold 튜닝 + StratifiedKFold 적용 예정.

---

## 5. Defect 역추적 결과 (핵심 센서)

XGBoost Feature Importance 기반 수율 불량 지배 센서 Top 5:

| 순위 | 센서 ID | Importance Score | 비고 |
|------|---------|----------------|------|
| 1 | Sensor_448 | 0.054 | 불량 발생과 가장 강한 연관성 |
| 2 | Sensor_196 | 0.048 | 1위와 유사한 수준 |
| 3 | Sensor_430 | 0.031 | - |
| 4 | Sensor_31  | 0.030 | - |
| 5 | Sensor_486 | 0.029 | - |

> 실제 Fab이라면 이 센서들을 우선 점검 → 공정 파라미터 이상 여부 확인하는 순서로 트러블슈팅 진행.

---

## 6. Tech Stack

| 구분 | 내용 |
|------|------|
| Language | Python 3.11 |
| Analysis | Pandas, NumPy, Scikit-learn |
| Modeling | XGBoost, Imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn |
| Environment | VS Code + Jupyter Notebook |

---

## 7. 파일 구조

```
wafer-sensor-defect-yield-prediction/
├── secom_analysis.ipynb      # 메인 분석 노트북
├── data/
│   └── uci-secom.csv         # SECOM 원본 데이터
├── eda_class_distribution.png
├── eda_null_distribution.png
├── feature_importance.png
├── model_evaluation.png
├── defect_traceback_boxplot.png
└── defect_scatter_matrix.png
```

---

## 8. 실행 방법

```bash
# 패키지 설치
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

# VS Code에서 노트북 실행
# secom_analysis.ipynb 열고 셀 순서대로 실행 (Shift+Enter)
```

---

## 9. 다음 단계 (개선 계획)

- [ ] LightGBM 모델 비교 실험
- [ ] Prediction Threshold 튜닝으로 Recall/Precision 트레이드오프 최적화
- [ ] SPC 관리도(Control Chart) 연동 — 핵심 센서 실시간 모니터링
- [ ] Streamlit 기반 공정 이상 감지 대시보드 구현
