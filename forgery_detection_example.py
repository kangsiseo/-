#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
인상파 화가 + 가품 검증 데이터셋 사용 예제
진품 vs 가품 분류 및 화가별 분류를 모두 지원합니다.
"""

from impressionist_forgery_dataset import (
    load_impressionist_data, 
    load_forgery_detection_data, 
    get_class_names
)
import numpy as np

def main():
    print("=" * 70)
    print("인상파 화가 + 가품 검증 데이터셋 사용 예제")
    print("=" * 70)
    
    # 1. 가품 검증용 데이터 (진품 vs 가품, 2클래스)
    print("\n🔍 1. 가품 검증용 데이터셋 로드 (진품 vs 가품)")
    print("-" * 50)
    (X_train_forgery, y_train_forgery), (X_test_forgery, y_test_forgery) = load_forgery_detection_data()
    
    print("\n📊 가품 검증 데이터셋 정보:")
    print(f"   X_train shape: {X_train_forgery.shape}")
    print(f"   y_train shape: {y_train_forgery.shape}")
    print(f"   X_test shape: {X_test_forgery.shape}")
    print(f"   y_test shape: {y_test_forgery.shape}")
    print(f"   클래스: 0=진품, 1=가품")
    
    # 2. 화가 분류용 데이터 (5클래스: 4명 화가 + 가품)
    print("\n🎨 2. 화가 분류용 데이터셋 로드 (4명 화가 + 가품)")
    print("-" * 50)
    (X_train_artist, y_train_artist), (X_test_artist, y_test_artist) = load_impressionist_data(include_fake=True)
    
    artist_class_names = get_class_names(include_fake=True)
    print("\n📊 화가 분류 데이터셋 정보:")
    print(f"   X_train shape: {X_train_artist.shape}")
    print(f"   y_train shape: {y_train_artist.shape}")
    print(f"   X_test shape: {X_test_artist.shape}")
    print(f"   y_test shape: {y_test_artist.shape}")
    print(f"   클래스 수: {len(artist_class_names)}")
    print(f"   클래스 이름: {artist_class_names}")
    
    # 3. 진품만 로드 (가품 제외)
    print("\n🖼️  3. 진품만 로드 (가품 제외)")
    print("-" * 50)
    (X_train_real, y_train_real), (X_test_real, y_test_real) = load_impressionist_data(include_fake=False)
    
    real_class_names = get_class_names(include_fake=False)
    print("\n📊 진품 전용 데이터셋 정보:")
    print(f"   X_train shape: {X_train_real.shape}")
    print(f"   클래스 이름: {real_class_names}")
    
    print("\n" + "=" * 70)
    print("🤖 머신러닝 모델 예제")
    print("=" * 70)
    
    # 4. 가품 검증 모델 예제
    print("\n1️⃣  가품 검증 모델 (Binary Classification)")
    print("-" * 40)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score
        
        # 데이터 평탄화
        X_train_flat = X_train_forgery.reshape(X_train_forgery.shape[0], -1)
        X_test_flat = X_test_forgery.reshape(X_test_forgery.shape[0], -1)
        
        # 모델 훈련
        print("   🔄 Random Forest 모델 훈련 중...")
        forgery_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        forgery_model.fit(X_train_flat, y_train_forgery)
        
        # 예측 및 평가
        y_pred_forgery = forgery_model.predict(X_test_flat)
        accuracy = accuracy_score(y_test_forgery, y_pred_forgery)
        
        print(f"   ✅ 가품 검증 정확도: {accuracy:.4f}")
        print("\n   📋 상세 분류 결과:")
        print(classification_report(y_test_forgery, y_pred_forgery, 
                                  target_names=['진품', '가품'], 
                                  digits=4))
        
    except ImportError:
        print("   ⚠️  scikit-learn이 설치되지 않아 모델 예제를 건너뜁니다.")
    
    # 5. 딥러닝 모델 예제 코드
    print("\n2️⃣  딥러닝 모델 예제 코드")
    print("-" * 40)
    
    cnn_example = '''
# TensorFlow/Keras를 사용한 CNN 모델 예제

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 가품 검증용 CNN 모델
def create_forgery_detection_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 이진 분류
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 화가 분류용 CNN 모델
def create_artist_classification_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')  # 5클래스 분류
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 모델 훈련 예제
# 가품 검증 모델
forgery_model = create_forgery_detection_model()
forgery_model.fit(X_train_forgery, y_train_forgery,
                  batch_size=32,
                  epochs=20,
                  validation_data=(X_test_forgery, y_test_forgery))

# 화가 분류 모델
artist_model = create_artist_classification_model()
y_train_artist_onehot = to_categorical(y_train_artist, 5)
y_test_artist_onehot = to_categorical(y_test_artist, 5)

artist_model.fit(X_train_artist, y_train_artist_onehot,
                 batch_size=32,
                 epochs=20,
                 validation_data=(X_test_artist, y_test_artist_onehot))
'''
    
    print(cnn_example)
    
    print("\n" + "=" * 70)
    print("🎯 사용 가능한 기능 요약")
    print("=" * 70)
    print("1. load_forgery_detection_data() - 가품 검증용 (진품 vs 가품)")
    print("2. load_impressionist_data(include_fake=True) - 화가 분류 + 가품")
    print("3. load_impressionist_data(include_fake=False) - 화가 분류만")
    print("4. get_class_names(include_fake=True/False) - 클래스 이름 반환")
    print("\n💡 전문적인 가품 검증 시스템을 구축할 수 있습니다!")
    print("=" * 70)

if __name__ == "__main__":
    main()
