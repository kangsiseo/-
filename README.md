# 인상파 화가 데이터셋 (Impressionist Painters Dataset)

이 데이터셋은 4명의 대표적인 인상파 화가들의 작품을 포함합니다.

## 포함된 화가들

- **Claude Monet** (73개 작품) - 인상파의 아버지, "인상, 해돋이"로 인상파라는 명칭의 기원
- **Pierre-Auguste Renoir** (230개 작품) - 인상파의 대표 화가 중 한 명
- **Camille Pissarro** (91개 작품) - 인상파의 아버지 격 존재, 유일하게 8회 인상파 전시에 모두 참여
- **Alfred Sisley** (259개 작품) - 순수 인상파 화가, 풍경화 전문

## 총 작품 수
653개의 인상파 작품

## 사용법

```python
from impressionist_dataset import load_impressionist_data

# MNIST 스타일로 데이터 로드
(X_train, y_train), (X_test, y_test) = load_impressionist_data()

# 또는 모든 데이터 로드
X, y = load_impressionist_data(split=False)
```

## 클래스 라벨
- 0: Alfred Sisley
- 1: Camille Pissarro  
- 2: Claude Monet
- 3: Pierre-Auguste Renoir

## 라이선스
교육 및 연구 목적으로 사용 가능합니다.
