# 인상파 화가 + 가품 검증 데이터셋 (Impressionist Painters + Forgery Detection Dataset)

이 데이터셋은 4명의 대표적인 인상파 화가들의 진품과 인상주의 가품을 포함하여 **가품 검증 모델 개발**에 특화된 데이터셋입니다.

## 📊 데이터 구성

### 진품 (Real Artworks)
- **Claude Monet** (73개 작품) - 인상파의 아버지, "인상, 해돋이"로 인상파라는 명칭의 기원
- **Pierre-Auguste Renoir** (230개 작품) - 인상파의 대표 화가 중 한 명
- **Camille Pissarro** (91개 작품) - 인상파의 아버지 격 존재, 유일하게 8회 인상파 전시에 모두 참여
- **Alfred Sisley** (259개 작품) - 순수 인상파 화가, 풍경화 전문

### 가품 (Fake Artworks)
- **인상파가품** (191개 작품) - AI 생성 및 인상주의 스타일 모방 작품

## 🎯 총 작품 수
- **진품**: 653개
- **가품**: 191개
- **전체**: 844개 작품

## 🚀 사용법

### 1. 가품 검증용 (진품 vs 가품, 2클래스)
```python
from impressionist_forgery_dataset import load_forgery_detection_data

# 가품 검증용 데이터 로드
(X_train, y_train), (X_test, y_test) = load_forgery_detection_data()

# 라벨: 0=진품, 1=가품
```

### 2. 화가 분류 + 가품 포함 (5클래스)
```python
from impressionist_forgery_dataset import load_impressionist_data

# 화가별 분류 + 가품 포함
(X_train, y_train), (X_test, y_test) = load_impressionist_data(include_fake=True)

# 라벨: 0=Sisley, 1=Pissarro, 2=Monet, 3=Renoir, 4=가품
```

### 3. 화가 분류만 (가품 제외, 4클래스)
```python
# 진품만 화가별 분류
(X_train, y_train), (X_test, y_test) = load_impressionist_data(include_fake=False)

# 라벨: 0=Sisley, 1=Pissarro, 2=Monet, 3=Renoir
```

## 🏷️ 클래스 라벨

### 가품 검증용 (Binary Classification)
- **0**: 진품 (Real)
- **1**: 가품 (Fake)

### 화가 분류용 (Multi-class Classification)
- **0**: Alfred Sisley
- **1**: Camille Pissarro  
- **2**: Claude Monet
- **3**: Pierre-Auguste Renoir
- **4**: 가품 (Fake) - `include_fake=True`일 때만

## 🔬 활용 분야
- **미술품 가품 검증 시스템**
- **컴퓨터 비전 기반 예술 작품 분석**
- **인상파 화가 스타일 분류**
- **딥러닝 모델 벤치마킹**
- **예술사 연구 및 교육**

## 📝 라이선스
교육 및 연구 목적으로 사용 가능합니다.
