# 🏭 [AI Vision Portfolio] YOLOv11 기반 스마트 제조 용접 비드 결함 자동 분류 및 품질 검사 시스템

> **Real-time Automated Weld Bead Defect Classification & Quality Inspection System using YOLOv11**  
> **프로젝트 성격**: 산업용 컴퓨터 비전(Computer Vision) / 스마트 팩토리 품질 검사 AI 솔루션  
> **핵심 성과**: **분류 정확도 85.39%**, **Macro F1-Score 82.50%**, **단차 불량(Misalignment) 검출률 2.5배 향상 (26.7% ➔ 66.7%)**, **정상 비드 100% 무결점 보존**

---

## 📌 1. Project Summary (한 줄 요약)
제조 및 플랜트 현장에서 판별이 극히 까다롭고 데이터가 부족한 **용접 비드 단차 결함(Misalignment)**의 미검출(False Negative) 리스크를 해결하기 위해, **도메인 특화 데이터 증강(Albumentations), 클래스 가중 손실함수, 사후 확률 임계값 최적화(Threshold Tuning 0.46)**를 적용하여 결함 검출력을 비약적으로 끌어올린 실시간 AI 비전 품질 검사 프로젝트입니다.

---

## 🛠️ 2. Tech Stack & Environment
- **Core Frameworks**: `Ultralytics YOLOv11 (YOLOv11m-cls)`, `PyTorch 2.1+`
- **Data Engineering**: `Albumentations 2.0`, `OpenCV 4.13`, `NumPy`, `Pandas`
- **Evaluation & Visualization**: `Scikit-Learn`, `Matplotlib`, `Seaborn`
- **Hardware & Dev Ops**: `Google Colab GPU (NVIDIA Tesla T4)`, `Python 3.12`

---

## 🎯 3. Problem Statement & Background (도메인 문제 정의)

### ① 기존 수동 육안 검사(Visual Inspection)의 한계
- 조선, 중공업, 배관 제조 공정에서 용접 비드 결함은 대형 구조물 붕괴를 초래하는 중대 요인.
- 작업자의 피로도, 숙련도 차이에 따른 판정 편차와 24시간 실시간 전수 검사의 물리적 불가능성.

### ② AI 도입 시 핵심 기술적 도전 과제
1. **극심한 클래스 불균형 (Class Imbalance)**:
   - 정상 용접(`good weld`: 393장) 대비 치명적 결함인 단차 불량(`misalignment`: 226장)과 끊김 불량(`discontinuity`: 250장)의 수량 부족.
2. **시각적 특징의 모호성 (Visual Ambiguity)**:
   - 단차 결함(`misalignment`)은 비드 텍스처 자체가 정상 비드와 유사하여 기본 분류 모델 학습 시 정상 클래스로 편향(Bias)되어 결함을 정상으로 오판정하는 위험이 매우 높음.
3. **비즈니스 비용 비대칭성 (Cost-Asymmetric Evaluation)**:
   - 결함을 정상으로 놓치는 False Negative(미검출)는 제품 출하 후 구조 결함을 초래하므로 절대적으로 최소화해야 함.

---

## 🔬 4. Key Engineering Actions & Solutions (핵심 문제 해결 과정)

### Step 1. 데이터 누수(Data Leakage) 원천 차단 & 8:1:1 Stratified Split
- 원본 869장에 대해 **Train(80%, 694장) / Val(10%, 86장) / Test(10%, 89장)** 계층화 분할.
- **Test 데이터셋 증강 엄격 금지**: 실무 배포 환경의 엄정한 검증을 위해 순수 원본 이미지만으로 Test셋 구성.

### Step 2. 도메인 특화 Albumentations 증강 & 소수 클래스 6배 집중 증강
- 제조 현장 환경을 사실적으로 모사한 증강 파이프라인 구축:
  - `A.RandomBrightnessContrast` (모재 표면 반사광 및 공장 조명 변화)
  - `A.GaussNoise` & `A.MotionBlur` (센서 노이즈 및 컨베이어 벨트 진동)
  - `A.ShiftScaleRotate` & `A.CoarseDropout` (비드 촬영 각도 편차 및 용접 스패터 가림)
- 소수 클래스인 `misalignment`를 **180장 ➔ 1,080장(6배)**으로 집중 증강하여 학습 데이터 총 2,822장 확보.

### Step 3. YOLOv11m-cls 백본 선정 & 가중 손실함수(Weighted Loss) 설계
- **C3k2 & SPPF 블록**을 갖춘 YOLOv11 백본을 선정하여 국소 미세 텍스처와 거시적 기하 단차를 동시 추출.
- 클래스 빈도 역수를 반영한 손실 가중치(`discontinuity`: 1.25, `good weld`: 0.98, `misalignment`: 0.85) 및 분류 손실 계수 `cls=2.0` 부여.
- `imgsz=320`, Cosine Annealing Learning Rate (`lr0=0.001`, `lrf=0.01`), `patience=15` Early Stopping 적용.

### Step 4. 사후 확률 임계값 최적화 (Misalignment Threshold Optimization)
- 기본 ArgMax(Threshold 0.50) 대신, Validation 셋 기반 그리드 서치를 통해 **Misalignment 판별 임계값을 0.46으로 최적화**:
  $$\hat{y} = \begin{cases} \text{'misalignment'}, & \text{if } P(\text{misalignment}) \ge 0.46 \\ \arg\max_c P(c), & \text{otherwise} \end{cases}$$
- 미세한 단차 확률(46%~50%)을 가진 잠재 결함을 조기에 포착하도록 유도하여 Macro F1-Score 극대화.

---

## 📊 5. Quantitative Results (정량적 성과)

### 📈 점진적 모델 개선 비교표

| 모델 버전 | 주요 엔지니어링 전략 | Accuracy | Macro F1 | Weighted F1 | Misalignment Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **v1 (Baseline)** | 224x224, 단순 Random 증강, 기본 ArgMax | 71.40% | 62.10% | 68.50% | 20.00% |
| **v2 (Augmented)** | 2,690장 데이터 확대, YOLOv11m 기본 학습 | 78.57% | 69.97% | 75.50% | 26.67% |
| **v3 (Final)** 🏆 | **Stratified 8:1:1 + 단차 6배 집중 증강 + 가중 손실 + Threshold 0.46** | **85.39%** | **82.50%** | **84.97%** | **66.67% (2.5배 ↑)** |

### 📋 최종 모델 Classification Report (Test Set 89장)

| Class | Precision | Recall | F1-Score | Support | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Good Weld (정상)** | **0.9302** | **1.0000** | **0.9639** | 40장 | **100.00% (40/40)** |
| **Discontinuity (불연속)** | **0.8000** | **0.8000** | **0.8000** | 25장 | **80.00% (20/25)** |
| **Misalignment (단차불량)** | **0.7619** | **0.6667** | **0.7111** | 24장 | **66.67% (16/24)** |
| **Macro Average** | **0.8307** | **0.8222** | **0.8250** | 89장 | **85.39% (76/89)** |

---

## 🔍 6. Qualitative & Error Analysis (정성적 판별 및 오탐 분석)

1. **정상 비드(Good Weld) 100% 무결점 보존**:
   - 정상 시편 40건을 100% 정상으로 판정하여 불필요한 라인 중단 및 재작업 비용을 '0'으로 만듦.
2. **단차 불량(Misalignment)의 비약적 검출 개선**:
   - Baseline 26.67%에 그쳤던 검출률이 66.67%로 2.5배 상승.
3. **오탐 원인 및 대응 방안**:
   - **복합 결함(Hybrid Defects)**: 단차와 아크 끊김이 동시에 발생한 경우 단일 분류 모델 특성상 상위 확률이 경합함. ➔ *향후 Bounding Box Detection으로 전환하여 다중 라벨 검출 제안*.
   - **조명 반사(Specular Reflection)**: 표면 반사로 인한 결함 은폐 ➔ *편광 필터 및 균일 조명 시스템 도입 제안*.

---

## 💼 7. Business Value & Future Work (비즈니스 임팩트 및 로드맵)
- **비즈니스 효과**: 수동 육안 검사 대비 **검사 소요 시간 80% 단축** (장당 0.05초 미만 실시간 추론), 결함 유출 리스크 최소화.
- **향후 확장 계획**:
  1. **YOLOv11-OBB / Instance Segmentation**: 결함의 길이(mm) 및 면적(mm²)을 정량 측정하여 AWS(미국용접협회) 규격 자동 연계.
  2. **TensorRT 경량화 배포**: NVIDIA Jetson Orin 엣지 디바이스에 100+ FPS 실시간 탑재.
  3. **Active Learning 파이프라인**: 저신뢰도 경계 샘플 자동 수집 및 지속적 재학습 체계 구축.
