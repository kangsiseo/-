    X_real = np.vstack(real_images) if real_images else np.empty((0, 224, 224, 3))
    y_real = np.hstack(real_labels) if real_labels else np.empty((0,))
    
    # 가품 데이터 로드
    fake_folder = dataset_path / "인상파가품"
    X_fake, y_fake = load_images_from_folder(fake_folder, 1)  # 가품: 라벨 1
    
    # 데이터 합치기
    X = np.vstack([X_real, X_fake])
    y = np.hstack([y_real, y_fake])
    
    print(f"✅ 데이터 로딩 완료!")
    print(f"   진품: {len(y_real)}개")
    print(f"   가품: {len(y_fake)}개")
    print(f"   총 {len(y)}개 이미지")
    
    # 데이터 셔플 및 분할
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"🔄 데이터 분할 완료:")
    print(f"   훈련 세트: {len(X_train)}개 (진품: {sum(y_train==0)}, 가품: {sum(y_train==1)})")
    print(f"   테스트 세트: {len(X_test)}개 (진품: {sum(y_test==0)}, 가품: {sum(y_test==1)})")
    
    return (X_train, y_train), (X_test, y_test)

def load_impressionist_data(include_fake=False, test_size=0.2, random_state=42, cache_dir="~/.impressionist_dataset"):
    """
    인상파 화가 분류용 데이터를 로드합니다
    
    Args:
        include_fake (bool): 가품을 포함할지 여부
        test_size (float): 테스트 세트 비율
        random_state (int): 랜덤 시드
        cache_dir (str): 캐시 디렉토리
    
    Returns:
        ((X_train, y_train), (X_test, y_test)):
        - X: (samples, 224, 224, 3) 이미지 데이터
        - y: 화가별 라벨 (include_fake=True면 4번이 가품)
    """
    # 데이터셋 다운로드
    dataset_dir = download_dataset(cache_dir)
    dataset_path = Path(dataset_dir)
    
    if include_fake:
        print("🎨 인상파 화가 + 가품 분류 데이터셋 로딩 중...")
    else:
        print("🎨 인상파 화가 분류 데이터셋 로딩 중...")
    
    # 화가별 데이터 로드
    all_images = []
    all_labels = []
    
    artists = ['Alfred_Sisley', 'Camille_Pissarro', 'Claude_Monet', 'Pierre-Auguste_Renoir']
    
    for i, artist in enumerate(artists):
        artist_folder = dataset_path / "인상파_Claude_Monet_Pierre-Auguste_Renoir_Camille_Pissarro_Alfred_Sisley" / artist
        if artist_folder.exists():
            imgs, lbls = load_images_from_folder(artist_folder, i)
            all_images.append(imgs)
            all_labels.append(lbls)
    
    # 가품 데이터 추가 (옵션)
    if include_fake:
        fake_folder = dataset_path / "인상파가품"
        if fake_folder.exists():
            imgs, lbls = load_images_from_folder(fake_folder, 4)  # 가품: 라벨 4
            all_images.append(imgs)
            all_labels.append(lbls)
    
    # 데이터 합치기
    X = np.vstack(all_images) if all_images else np.empty((0, 224, 224, 3))
    y = np.hstack(all_labels) if all_labels else np.empty((0,))
    
    print(f"✅ 데이터 로딩 완료!")
    class_names = get_class_names(include_fake)
    for i, name in enumerate(class_names):
        count = sum(y == i)
        print(f"   {i}: {name} - {count}개")
    print(f"   총 {len(y)}개 이미지")
    
    # 데이터 셔플 및 분할
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"🔄 데이터 분할 완료:")
    print(f"   훈련 세트: {len(X_train)}개")
    print(f"   테스트 세트: {len(X_test)}개")
    
    return (X_train, y_train), (X_test, y_test)

def get_sample_images(num_samples=5, include_fake=False, cache_dir="~/.impressionist_dataset"):
    """
    각 클래스별로 샘플 이미지를 반환합니다
    
    Args:
        num_samples (int): 클래스별 샘플 개수
        include_fake (bool): 가품 포함 여부
        cache_dir (str): 캐시 디렉토리
    
    Returns:
        dict: {class_name: [image_arrays]}
    """
    # 데이터셋 다운로드
    dataset_dir = download_dataset(cache_dir)
    dataset_path = Path(dataset_dir)
    
    samples = {}
    
    # 화가별 샘플
    artists = {
        'Alfred_Sisley': 'Alfred Sisley',
        'Camille_Pissarro': 'Camille Pissarro', 
        'Claude_Monet': 'Claude Monet',
        'Pierre-Auguste_Renoir': 'Pierre-Auguste Renoir'
    }
    
    for folder_name, display_name in artists.items():
        artist_folder = dataset_path / "인상파_Claude_Monet_Pierre-Auguste_Renoir_Camille_Pissarro_Alfred_Sisley" / folder_name
        if artist_folder.exists():
            imgs, _ = load_images_from_folder(artist_folder, 0, max_count=num_samples)
            samples[display_name] = imgs
    
    # 가품 샘플 (옵션)
    if include_fake:
        fake_folder = dataset_path / "인상파가품"
        if fake_folder.exists():
            imgs, _ = load_images_from_folder(fake_folder, 1, max_count=num_samples)
            samples['가품 (Fake)'] = imgs
    
    return samples

def visualize_samples(num_samples=3, include_fake=False, figsize=(15, 10)):
    """
    각 클래스별 샘플 이미지를 시각화합니다
    
    Args:
        num_samples (int): 클래스별 샘플 개수
        include_fake (bool): 가품 포함 여부
        figsize (tuple): 그림 크기
    """
    try:
        import matplotlib.pyplot as plt
        
        # 샘플 이미지 가져오기
        samples = get_sample_images(num_samples, include_fake)
        
        # 플롯 설정
        num_classes = len(samples)
        fig, axes = plt.subplots(num_classes, num_samples, figsize=figsize)
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        elif num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        # 각 클래스별 샘플 표시
        for i, (class_name, images) in enumerate(samples.items()):
            for j in range(min(num_samples, len(images))):
                ax = axes[i, j] if num_samples > 1 else axes[i]
                ax.imshow(images[j])
                ax.set_title(f"{class_name}", fontsize=10)
                ax.axis('off')
            
            # 빈 subplot 숨기기
            for j in range(len(images), num_samples):
                if num_samples > 1:
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.suptitle('인상파 가품 검증 데이터셋 샘플', fontsize=16, y=1.02)
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib이 설치되어 있지 않아 시각화를 건너뜁니다.")
        print("설치: pip install matplotlib")

# 편의 함수들
def print_dataset_info():
    """데이터셋 정보를 출력합니다"""
    print("🎨 인상파 가품 검증 데이터셋 v2.0")
    print("=" * 50)
    print("📊 구성:")
    print("   • 진품: 653개 (4명 인상파 화가)")
    print("     - Alfred Sisley: 259개")  
    print("     - Pierre-Auguste Renoir: 230개")
    print("     - Camille Pissarro: 91개")
    print("     - Claude Monet: 73개")
    print("   • 가품: 653개 (AI 생성 + 데이터 증강)")
    print("   • 총 1,306개 이미지")
    print()
    print("🚀 사용법:")
    print("   1. 가품 검증 (2클래스):")
    print("      (X_train, y_train), (X_test, y_test) = load_forgery_detection_data()")
    print()
    print("   2. 화가 + 가품 분류 (5클래스):")
    print("      (X_train, y_train), (X_test, y_test) = load_impressionist_data(include_fake=True)")
    print()
    print("   3. 화가만 분류 (4클래스):")
    print("      (X_train, y_train), (X_test, y_test) = load_impressionist_data(include_fake=False)")
    print()
    print("🎯 특징:")
    print("   • MNIST 스타일 API")
    print("   • 자동 다운로드 & 캐싱")
    print("   • 224x224 RGB 이미지")
    print("   • 정규화된 픽셀 값 (0-1)")
    print("   • 균형 잡힌 데이터셋")

if __name__ == "__main__":
    # 데이터셋 정보 출력
    print_dataset_info()
    
    # 샘플 로드 테스트
    print("\n🧪 샘플 데이터 로드 테스트:")
    try:
        (X_train, y_train), (X_test, y_test) = load_forgery_detection_data()
        print("✅ 가품 검증 데이터 로드 성공!")
        
        (X_train2, y_train2), (X_test2, y_test2) = load_impressionist_data(include_fake=True)
        print("✅ 화가 + 가품 분류 데이터 로드 성공!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
