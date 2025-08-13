#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
인상파 화가 데이터셋 사용 예제
MNIST 스타일로 데이터를 로드하고 기본 분석을 수행합니다.
"""

from impressionist_dataset import load_impressionist_data, get_class_names
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 60)
    print("인상파 화가 데이터셋 로드 예제")
    print("=" * 60)
    
    # 1. MNIST 스타일로 데이터 로드
    print("\n1. 데이터 로드 중...")
    (X_train, y_train), (X_test, y_test) = load_impressionist_data()
    
    # 2. 데이터 정보 출력
    print("\n2. 데이터셋 정보:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_test shape: {y_test.shape}")
    
    # 3. 클래스 정보
    class_names = get_class_names()
    print(f"\n3. 클래스 정보:")
    print(f"   클래스 수: {len(class_names)}")
    print(f"   클래스 이름: {class_names}")
    
    # 4. 라벨 분포
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n4. 훈련 데이터 라벨 분포:")
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"   {label}: {class_names[label]} - {count}개")
    
    # 5. 샘플 이미지 표시 (matplotlib이 사용 가능한 환경에서)
    try:
        print("\n5. 샘플 이미지 저장 중...")
        
        plt.figure(figsize=(15, 12))
        
        # 각 클래스별로 3개씩 샘플 표시
        for class_idx in range(len(class_names)):
            # 해당 클래스의 이미지 찾기
            class_indices = np.where(y_train == class_idx)[0]
            
            # 처음 3개 선택
            for i in range(min(3, len(class_indices))):
                plt.subplot(len(class_names), 3, class_idx * 3 + i + 1)
                
                img_idx = class_indices[i]
                plt.imshow(X_train[img_idx])
                plt.title(f'{class_names[class_idx]}')
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
        print("   샘플 이미지가 'sample_images.png'로 저장되었습니다.")
        
    except ImportError:
        print("   matplotlib이 설치되지 않아 이미지 표시를 건너뜁니다.")
    except Exception as e:
        print(f"   이미지 저장 중 오류: {e}")
    
    print("\n" + "=" * 60)
    print("데이터 로드 완료! 이제 머신러닝 모델 훈련에 사용할 수 있습니다.")
    print("=" * 60)
    
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    main()
