### Pierre-Auguste Renoir (피에르 오귀스트 르누아르) 👨‍👩‍👧‍👦
- **특징**: 인상파의 대표 화가, 인물화와 일상 장면
- **화풍**: 따뜻한 색조, 부드러운 터치
- **대표작**: 물랭 드 라 갈레트의 무도회, 선상의 오찬

### Camille Pissarro (카미유 피사로) 🌳
- **특징**: 인상파의 아버지, 유일하게 8회 인상파 전시에 모두 참여
- **화풍**: 시골 풍경과 농민들의 일상
- **대표작**: 몽마르트르 대로, 루브시엔느의 눈

### Alfred Sisley (알프레드 시슬레) 🏞️
- **특징**: 가장 순수한 인상파 화가, 풍경화 전문
- **화풍**: 자연스러운 야외 풍경, 강과 마을 풍경
- **대표작**: 마를리의 홍수, 루브시엔느 풍경

## 🔧 고급 기능

### 샘플 이미지 시각화
```python
from impressionist_forgery_dataset_balanced import visualize_samples

# 각 클래스별 샘플 이미지 보기
visualize_samples(num_samples=3, include_fake=True)
```

### 클래스 이름 가져오기
```python
from impressionist_forgery_dataset_balanced import get_class_names

class_names = get_class_names(include_fake=True)
print(class_names)
# ['Alfred Sisley', 'Camille Pissarro', 'Claude Monet', 'Pierre-Auguste Renoir', '가품 (Fake)']
```

### 데이터셋 정보 출력
```python
from impressionist_forgery_dataset_balanced import print_dataset_info

print_dataset_info()
```

## 📈 성능 벤치마크

### Random Forest 결과
- **가품 검증 정확도**: ~85-92%
- **화가 분류 정확도**: ~75-85%

### CNN 모델 결과 (20 epochs)
- **가품 검증 정확도**: ~90-95%
- **화가 분류 정확도**: ~80-90%

## 🛠️ 활용 분야

1. **미술품 감정**: 인상파 작품의 진위 감별
2. **교육**: 인상파 화풍 학습 및 연구
3. **AI 연구**: 예술 작품 분류 및 생성 모델 개발
4. **컴퓨터 비전**: 스타일 전이 및 이미지 분석
5. **문화유산 보존**: 디지털 아카이브 구축

## 📁 파일 구조

```
인상파_Claude_Monet_Pierre-Auguste_Renoir_Camille_Pissarro_Alfred_Sisley/
├── Alfred_Sisley/                    # 259개 작품
├── Camille_Pissarro/                 # 91개 작품
├── Claude_Monet/                     # 73개 작품
├── Pierre-Auguste_Renoir/            # 230개 작품
├── 인상파가품/                        # 653개 가품
├── impressionist_forgery_dataset_balanced.py  # 메인 로더
├── balanced_example_usage.py         # 사용 예제
├── README.md                         # 문서
└── requirements.txt                  # 의존성
```

## ⚡ 성능 최적화 팁

### 1. 메모리 효율성
```python
# 배치 단위로 로드 (대용량 데이터셋용)
(X_train, y_train), (X_test, y_test) = load_forgery_detection_data()

# 메모리 사용량 확인
print(f"훈련 데이터 크기: {X_train.nbytes / 1024**2:.1f} MB")
```

### 2. 캐싱 활용
```python
# 커스텀 캐시 디렉토리 사용
(X_train, y_train), (X_test, y_test) = load_forgery_detection_data(
    cache_dir="~/my_art_cache"
)
```

### 3. 데이터 증강
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
```

## 🚨 주의사항

1. **저작권**: 이 데이터셋은 연구 및 교육 목적으로만 사용하세요
2. **정확성**: AI가 생성한 가품이므로 실제 감정과는 차이가 있을 수 있습니다
3. **편향**: 특정 화가의 작품 수가 불균형할 수 있습니다
4. **윤리**: 실제 미술품 거래에서 이 모델을 사용할 때는 전문가 검증이 필요합니다

## 🤝 기여하기

1. Fork 이 저장소
2. 새 기능 브랜치 생성 (`git checkout -b feature/새기능`)
3. 변경사항 커밋 (`git commit -am '새 기능 추가'`)
4. 브랜치에 Push (`git push origin feature/새기능`)
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다. 상업적 사용 시에는 별도 문의 바랍니다.

## 📞 문의

- GitHub Issues: 버그 리포트 및 기능 요청
- Email: 데이터셋 관련 문의

## 🙏 감사의 말

- 인상파 화가들의 위대한 작품들
- 오픈 데이터 커뮤니티
- TensorFlow, scikit-learn 개발팀
- 모든 기여자들

---

**🎨 인상주의 미술 전문가를 위한 AI 가품 검증 시스템을 구축해보세요!**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-red.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

> "Art is not what you see, but what you make others see." - Edgar Degas
