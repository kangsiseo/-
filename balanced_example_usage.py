#!/usr/bin/env python3
"""
인상파 가품 검증 데이터셋 사용 예제 (균형 잡힌 버전)
진품 653개 + 가품 653개 = 총 1,306개 이미지
"""

# 라이브러리 설치가 필요한 경우:
# pip install numpy Pillow requests scikit-learn matplotlib tensorflow

from impressionist_forgery_dataset_balanced import (
    load_forgery_detection_data, 
    load_impressionist_data, 
    get_class_names,
    print_dataset_info,
    visualize_samples
)

def main():
    """메인 실행 함수"""
    
    # 데이터셋 정보 출력
    print_dataset_info()
    
    print("\n" + "="*60)
    print("🔍 1. 가품 검증 모델 (진품 vs 가품, 2클래스)")
    print("="*60)
    
    # 가품 검증용 데이터 로드
    (X_train, y_train), (X_test, y_test) = load_forgery_detection_data(random_state=42)
    
    print(f"\n📊 데이터 분포:")
    print(f"   훈련 세트: 진품 {sum(y_train==0)}개, 가품 {sum(y_train==1)}개")
    print(f"   테스트 세트: 진품 {sum(y_test==0)}개, 가품 {sum(y_test==1)}개")
    
    # Random Forest 모델로 가품 검증
    print(f"\n🤖 Random Forest 가품 검증 모델 훈련 중...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    
    # 데이터 평탄화 (Random Forest용)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 모델 훈련
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_flat, y_train)
    
    # 예측 및 평가
    y_pred = rf_model.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📈 Random Forest 결과:")
    print(f"   정확도: {accuracy:.4f}")
    print(f"\n📋 분류 리포트:")
    print(classification_report(y_test, y_pred, target_names=['진품', '가품']))
    
    print("\n" + "="*60)
    print("🎨 2. 화가 + 가품 분류 모델 (5클래스)")
    print("="*60)
    
    # 화가 + 가품 분류용 데이터 로드
    (X_train_art, y_train_art), (X_test_art, y_test_art) = load_impressionist_data(
        include_fake=True, random_state=42
    )
    
    class_names = get_class_names(include_fake=True)
    
    print(f"\n📊 클래스별 분포:")
    for i, name in enumerate(class_names):
        train_count = sum(y_train_art == i)
        test_count = sum(y_test_art == i)
        print(f"   {i}: {name} - 훈련: {train_count}개, 테스트: {test_count}개")
    
    # Random Forest로 화가 분류
    print(f"\n🎨 Random Forest 화가 분류 모델 훈련 중...")
    
    X_train_art_flat = X_train_art.reshape(X_train_art.shape[0], -1)
    X_test_art_flat = X_test_art.reshape(X_test_art.shape[0], -1)
    
    rf_artist_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_artist_model.fit(X_train_art_flat, y_train_art)
    
    y_pred_art = rf_artist_model.predict(X_test_art_flat)
    accuracy_art = accuracy_score(y_test_art, y_pred_art)
    
    print(f"\n📈 화가 분류 결과:")
    print(f"   정확도: {accuracy_art:.4f}")
    print(f"\n📋 분류 리포트:")
    print(classification_report(y_test_art, y_pred_art, target_names=class_names))
    
    print("\n" + "="*60)
    print("🧠 3. 딥러닝 CNN 모델 예제")
    print("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        print("🔥 TensorFlow CNN 가품 검증 모델 구성 중...")
        
        # CNN 모델 구성
        def create_forgery_cnn_model():
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Conv2D(256, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')  # 이진 분류
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        # 모델 생성
        cnn_model = create_forgery_cnn_model()
        print(cnn_model.summary())
        
        print(f"\n🚀 CNN 모델 훈련 시작 (빠른 테스트용 3 epoch)...")
        
        # 빠른 테스트를 위해 적은 epoch로 훈련
        history = cnn_model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=3,  # 실제 사용시에는 20-50으로 늘리세요
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # 평가
        test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
        print(f"\n🎯 CNN 최종 결과:")
        print(f"   테스트 정확도: {test_acc:.4f}")
        print(f"   테스트 손실: {test_loss:.4f}")
        
        # 예측 예제
        print(f"\n🔮 샘플 예측:")
        sample_predictions = cnn_model.predict(X_test[:5])
        for i, (pred, true) in enumerate(zip(sample_predictions, y_test[:5])):
            pred_class = "가품" if pred[0] > 0.5 else "진품"
            true_class = "가품" if true == 1 else "진품"
            confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
            print(f"   샘플 {i+1}: 예측={pred_class} ({confidence:.3f}), 실제={true_class}")
        
    except ImportError:
        print("❗ TensorFlow가 설치되어 있지 않습니다.")
        print("설치: pip install tensorflow")
    except Exception as e:
        print(f"❌ CNN 모델 실행 중 오류: {e}")
    
    print("\n" + "="*60)
    print("📊 4. 데이터 시각화")
    print("="*60)
    
    try:
        # 클래스별 샘플 이미지 시각화
        print("🖼️ 클래스별 샘플 이미지 시각화 중...")
        visualize_samples(num_samples=3, include_fake=True, figsize=(15, 12))
        
    except Exception as e:
        print(f"❌ 시각화 오류: {e}")
        print("matplotlib 설치 확인: pip install matplotlib")
    
    print("\n" + "="*60)
    print("✅ 모든 예제 실행 완료!")
    print("="*60)
    print("\n💡 추가 활용 팁:")
    print("1. 더 높은 정확도를 위해 CNN 모델의 epoch을 20-50으로 늘려보세요")
    print("2. 데이터 증강(augmentation)을 추가해보세요")
    print("3. 전이 학습(Transfer Learning)을 사용해보세요 (ResNet, VGG 등)")
    print("4. 앙상블 방법으로 여러 모델을 결합해보세요")
    print("5. 특성 중요도를 분석해서 어떤 부분이 가품 판별에 중요한지 확인해보세요")
    
    print(f"\n🎨 인상파 전문가를 위한 AI 가품 검증 시스템 구축 완료!")

if __name__ == "__main__":
    main()
