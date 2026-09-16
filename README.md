# 🏭 [WeldAI Vision] YOLOv11 기반 스마트 제조 용접 비드 결함 자동 검사 시스템

<div align="center">

[![Live Interactive Dashboard](https://img.shields.io/badge/🌐_Live_Dashboard-Click_to_Open-blue?style=for-the-badge&logo=google-chrome&logoColor=white)](https://jiu7021.github.io/weld/)
[![Open in Colab](https://img.shields.io/badge/Colab-Open_Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/jiu7021/weld/blob/main/weld.ipynb)
[![Ultralytics YOLOv11](https://img.shields.io/badge/Model-YOLOv11m--cls-00FFFF?style=for-the-badge&logo=yolo)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch_2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

<br>

**[👉 🌐 라이브 인터랙티브 시각화 대시보드 바로가기 (클릭)](https://jiu7021.github.io/weld/)**  
*(좌측 뷰어에서 3×3 검출 이미지를 클릭하면 우측 패널에서 모델의 시각적 판별 근거와 수치 해석이 실시간으로 연동됩니다)*

</div>

---

## 📊 1. Key Performance Highlights (핵심 성과 지표)

> **면접 및 발표용 핵심 수치 (소수점 1자리 기준)**

| 핵심 지표 (Metric) | Baseline (v1/v2) | Final Optimized (v3) | 개선 성과 (Improvement) | 핵심 엔지니어링 전략 |
| :--- | :---: | :---: | :---: | :--- |
| **전체 분류 정확도 (Accuracy)** | 78.6% | **85.4%** | **+6.8%p** 🚀 | Stratified 8:1:1 분할, imgsz=320, Cosine LR |
| **Macro F1-Score** | 70.0% | **82.5%** | **+12.5%p** 🔥 | 클래스 빈도 역수 가중치 손실함수 (`cls=2.0`) |
| **단차(Misalignment) Recall** | 26.7% | **66.7%** | **+40.0%p (2.5배 ↑)** 🎯 | **Misalignment Threshold 0.46 최적화** |
| **정상 비드(Good Weld) Accuracy** | 100.0% | **100.0%** | **오탐 제로 (Zero False Alarm)** ✨ | 40/40 전량 100% 정상 판정 (재작업 비용 0) |
| **비드 끊김(Discontinuity) Recall** | 93.3% | **80.0%** | **F1-Score 0.80 달성** 🛠️ | 끊김 및 모재 노출 구간 고신뢰도 식별 |
| **실시간 추론 속도** | - | **< 0.05초 / 장 (20+ FPS)** | **검사 시간 80% 단축** ⚡ | Tesla T4 Edge 실시간 전수 검사 지원 |

---

## 💡 2. Problem & Domain Background (문제 정의)

### ① 기존 수동 육안 검사의 한계점
- 자동차 차체(BIW), 샤시 및 배터리 팩 프레임 제조 현장에서 용접 비드 결함은 차량 비틀림 강성 저하 및 주행 진동 피로 파괴를 초래하는 치명적 요인.
- 검사관의 컨디션/피로도에 따른 판정 편차 및 24시간 실시간 전수 검사의 물리적 한계.

### ② AI 도입 시 3대 핵심 난제
1. **극심한 클래스 불균형**: 정상 비드(`good weld`: 393장) 대비 치명적 단차 결함(`misalignment`: 226장) 데이터 부족.
2. **시각적 특징의 모호성**: 단차 불량은 비드 텍스처 자체가 정상과 유사하여 일반 분류 모델 학습 시 정상 클래스로 편향(Bias)됨.
3. **비즈니스 비용 비대칭성**: 단차 결함을 정상으로 놓치는 **False Negative(미검출)**는 치명적이므로 결함 검출률(Recall) 극대화가 필수.

---

## 🔬 3. Core Solutions & Engineering Actions (핵심 문제 해결)

```
[1. 원본 869장] ➔ [2. Stratified 8:1:1 Split (Test 무증강 원칙)]
➔ [3. 소수 클래스 6배 집중 Albumentations 증강 (Train 2,822장)]
➔ [4. YOLOv11m-cls 백본 (C3k2/SPPF Feature Extraction, imgsz=320)]
➔ [5. Class-Weighted Loss (cls=2.0) + Cosine Annealing LR]
➔ [6. 사후 확률 임계값 최적화 (Misalignment Threshold = 0.46)]
➔ [7. 실시간 품질 판정: 100% 정상 보존 & 단차 검출률 66.7% 달성]
```

### 1) 데이터 누수(Data Leakage) 원천 차단 & 8:1:1 Stratified Split
- Train(80%, 694장) / Val(10%, 86장) / Test(10%, 89장) 분할.
- 실무 배포 신뢰성을 위해 **Test 데이터셋 증강을 엄격히 금지**하고 순수 원본으로만 공정 검증.

### 2) 도메인 특화 Albumentations 증강 & 소수 클래스 6배 집중 증강
- 공장 내 조명 반사, 센서 노이즈, 컨베이어 진동, 스패터 이물질을 가상 시뮬레이션.
- 소수 클래스인 `misalignment`를 **180장 ➔ 1,080장(6배)** 집중 증강하여 총 2,822장 학습 데이터 구축.

### 3) YOLOv11m-cls 백본 & 클래스 가중 손실함수 설계
- **C3k2 Block**(국소 비드 끊김 포착) + **SPPF Block**(거시적 단차 및 기울기 포착).
- 클래스 빈도 역수 가중치(`discontinuity`: 1.25, `good weld`: 0.98, `misalignment`: 0.85) 및 손실 계수 `cls=2.0` 적용.

### 4) 사후 확률 임계값 최적화 (Threshold Tuning)
- 기본 ArgMax(0.50) 대신, Validation 셋 그리드 서치를 통해 **Misalignment 판별 임계값을 0.46으로 튜닝**:
  $$\hat{y} = \begin{cases} \text{'misalignment'}, & \text{if } P(\text{misalignment}) \ge 0.46 \\ \arg\max_c P(c), & \text{otherwise} \end{cases}$$
- 잠재 단차 결함(46%~50% 확률)을 선제 포착하여 **단차 검출률을 기존 26.7%에서 66.7%로 2.5배 비약적 향상**.

---

## 📈 4. Confusion Matrix & Quantitative Report (정량 성과)

### 📋 최종 모델 Classification Report (Test 89장)

| Class | Precision | Recall | F1-Score | Support | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Good Weld (정상)** | **93.0%** | **100.0%** | **96.4%** | 40장 | **100.0% (40/40)** |
| **Discontinuity (끊김)** | **80.0%** | **80.0%** | **80.0%** | 25장 | **80.0% (20/25)** |
| **Misalignment (단차)** | **76.2%** | **66.7%** | **71.1%** | 24장 | **66.7% (16/24)** |
| **전체 평균 (Macro Avg)** | **83.1%** | **82.2%** | **82.5%** | 89장 | **85.4% (76/89)** |

---

## 🔍 5. Qualitative & Error Analysis (정성적 판별 및 오탐 분석)

1. **정상 비드(Good Weld) 100% 무결점 보존**:
   - 정상 40건을 100% 정상 판정하여 불필요한 라인 중단 및 재작업 비용을 '0'으로 만듦.
2. **단차 불량(Misalignment) 검출력 비약적 개선**:
   - Baseline 26.7%에 그쳤던 단차 결함 검출률을 66.7%로 대폭 개선.
3. **오탐(False Positive) 원인 및 개선 로드맵**:
   - **복합 결함(단차 + 비드 얇아짐)**: 두 클래스 특징이 공존하여 상위 확률 경합 ➔ *향후 YOLOv11 Bounding Box Detection으로 전환하여 다중 라벨 독립 검출 제안*.
   - **조명 반사**: 표면 광택으로 인한 결함 은폐 ➔ *편광 필터 및 균일 조명 시스템 도입 제안*.

---

## 🚀 6. Project Structure

```
.
├── index.html                   # 🌐 2-Pane 인터랙티브 포트폴리오 웹 대시보드
├── weld.ipynb                   # 📓 7단계 완성형 주피터 노트북 (Colab 실행 가능)
├── PORTFOLIO_WELD_INSPECTION.md # 📄 노션/이력서용 STAR 포트폴리오 기술서
├── assets/                      # 🖼️ EDA, 증강, 혼동행렬, 3x3 예측 시각화 이미지
└── .github/workflows/pages.yml  # ⚙️ GitHub Pages 자동 배포 워크플로우
```

---

<div align="center">
  <b>© 2026 Weld AI Vision Inspection Project. All Rights Reserved.</b>
</div>
