import os
import numpy as np
from PIL import Image
import requests
import zipfile
import tempfile
from sklearn.model_selection import train_test_split

def download_dataset():
    """GitHub에서 데이터셋을 다운로드합니다."""
    url = "https://github.com/kangsiseo/-/archive/refs/heads/main.zip"
    
    # 임시 디렉토리에 다운로드
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "dataset.zip")
        
        print("데이터셋을 다운로드하는 중...")
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 데이터 폴더 찾기
        extracted_dir = os.path.join(temp_dir, "--main")
        return extracted_dir

def load_images_from_folder(folder_path, label, target_size=(224, 224)):
    """폴더에서 이미지들을 로드하고 라벨을 할당합니다."""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"폴더를 찾을 수 없습니다: {folder_path}")
        return np.array([]), np.array([])
    
    # 지원하는 이미지 확장자
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    for filename in image_files:
        try:
            img_path = os.path.join(folder_path, filename)
            # 이미지 로드 및 리사이즈
            with Image.open(img_path) as img:
                # RGB 변환 (RGBA나 다른 모드를 RGB로 변환)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 리사이즈
                img = img.resize(target_size)
                
                # numpy 배열로 변환
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
                
        except Exception as e:
            print(f"이미지 로드 실패 {filename}: {e}")
            continue
    
    return np.array(images), np.array(labels)

def load_impressionist_data(include_fake=True, split=True, test_size=0.2, target_size=(224, 224), local_path=None):
    """
    인상파 화가 데이터셋을 로드합니다. (진품 + 가품 포함)
    
    Parameters:
    -----------
    include_fake : bool, default=True
        True면 가품 이미지도 포함, False면 진품만 포함
    split : bool, default=True
        True면 train/test로 분할, False면 전체 데이터 반환
    test_size : float, default=0.2
        테스트 데이터의 비율
    target_size : tuple, default=(224, 224)
        이미지 리사이즈 크기
    local_path : str, default=None
        로컬 데이터 경로 (None이면 GitHub에서 다운로드)
    
    Returns:
    --------
    if split=True:
        (X_train, y_train), (X_test, y_test)
    if split=False:
        X, y
        
    클래스 라벨:
    - 0: Alfred Sisley (진품)
    - 1: Camille Pissarro (진품)  
    - 2: Claude Monet (진품)
    - 3: Pierre-Auguste Renoir (진품)
    - 4: 가품 (Fake) - include_fake=True일 때만
    """
    
    # 화가별 라벨 매핑 (진품)
    artist_labels = {
        'Alfred_Sisley': 0,
        'Camille_Pissarro': 1,
        'Claude_Monet': 2,
        'Pierre-Auguste_Renoir': 3
    }
    
    # 데이터 경로 설정
    if local_path is None:
        data_dir = download_dataset()
    else:
        data_dir = local_path
    
    all_images = []
    all_labels = []
    
    print("진품 이미지를 로드하는 중...")
    
    # 각 화가별 이미지 로드 (진품)
    for artist_name, label in artist_labels.items():
        folder_path = os.path.join(data_dir, artist_name)
        images, labels = load_images_from_folder(folder_path, label, target_size)
        
        if len(images) > 0:
            all_images.extend(images)
            all_labels.extend(labels)
            print(f"  {artist_name}: {len(images)}개 진품 이미지 로드 완료")
        else:
            print(f"  {artist_name}: 이미지를 찾을 수 없습니다")
    
    # 가품 이미지 로드 (선택적)
    if include_fake:
        print("\n가품 이미지를 로드하는 중...")
        fake_folder_path = os.path.join(data_dir, '인상파가품')
        fake_images, fake_labels = load_images_from_folder(fake_folder_path, 4, target_size)  # 라벨 4 = 가품
        
        if len(fake_images) > 0:
            all_images.extend(fake_images)
            all_labels.extend(fake_labels)
            print(f"  가품: {len(fake_images)}개 이미지 로드 완료")
        else:
            print("  가품 이미지를 찾을 수 없습니다")
    
    # numpy 배열로 변환
    X = np.array(all_images)
    y = np.array(all_labels)
    
    print(f"\n총 {len(X)}개 이미지 로드 완료")
    print(f"이미지 shape: {X.shape}")
    
    # 라벨 분포 출력
    unique, counts = np.unique(y, return_counts=True)
    print(f"라벨 분포:")
    class_names = get_class_names(include_fake)
    for label, count in zip(unique, counts):
        print(f"  {label}: {class_names[label]} - {count}개")
    
    # 데이터 정규화 (0-1 범위)
    X = X.astype('float32') / 255.0
    
    if split:
        # train/test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]}개")
        print(f"Test set: {X_test.shape[0]}개")
        
        return (X_train, y_train), (X_test, y_test)
    else:
        return X, y

def get_class_names(include_fake=True):
    """클래스 이름을 반환합니다."""
    base_names = ['Alfred_Sisley', 'Camille_Pissarro', 'Claude_Monet', 'Pierre-Auguste_Renoir']
    if include_fake:
        return base_names + ['Fake_Artwork']
    return base_names

def load_forgery_detection_data(target_size=(224, 224), test_size=0.2, local_path=None):
    """
    가품 검증을 위한 이진 분류 데이터셋을 로드합니다.
    
    Returns:
    --------
    (X_train, y_train), (X_test, y_test)
    
    라벨:
    - 0: 진품 (Real)
    - 1: 가품 (Fake)
    """
    
    # 데이터 경로 설정
    if local_path is None:
        data_dir = download_dataset()
    else:
        data_dir = local_path
    
    all_images = []
    all_labels = []
    
    print("가품 검증용 데이터셋을 로드하는 중...")
    
    # 진품 로드 (모든 인상파 화가 통합)
    artist_folders = ['Alfred_Sisley', 'Camille_Pissarro', 'Claude_Monet', 'Pierre-Auguste_Renoir']
    real_count = 0
    
    for artist_name in artist_folders:
        folder_path = os.path.join(data_dir, artist_name)
        images, _ = load_images_from_folder(folder_path, 0, target_size)  # 라벨 0 = 진품
        
        if len(images) > 0:
            all_images.extend(images)
            all_labels.extend([0] * len(images))  # 모든 진품은 라벨 0
            real_count += len(images)
    
    print(f"  진품: {real_count}개 이미지 로드 완료")
    
    # 가품 로드
    fake_folder_path = os.path.join(data_dir, '인상파가품')
    fake_images, _ = load_images_from_folder(fake_folder_path, 1, target_size)  # 라벨 1 = 가품
    
    if len(fake_images) > 0:
        all_images.extend(fake_images)
        all_labels.extend([1] * len(fake_images))  # 모든 가품은 라벨 1
        print(f"  가품: {len(fake_images)}개 이미지 로드 완료")
    
    # numpy 배열로 변환
    X = np.array(all_images)
    y = np.array(all_labels)
    
    print(f"\n총 {len(X)}개 이미지 로드 완료")
    print(f"이미지 shape: {X.shape}")
    
    # 라벨 분포
    unique, counts = np.unique(y, return_counts=True)
    print(f"라벨 분포:")
    for label, count in zip(unique, counts):
        label_name = "진품" if label == 0 else "가품"
        print(f"  {label}: {label_name} - {count}개")
    
    # 데이터 정규화
    X = X.astype('float32') / 255.0
    
    # train/test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]}개")
    print(f"Test set: {X_test.shape[0]}개")
    
    return (X_train, y_train), (X_test, y_test)

# 사용 예제
if __name__ == "__main__":
    print("=" * 60)
    print("인상파 화가 + 가품 검증 데이터셋")
    print("=" * 60)
    
    # 1. 화가 분류용 데이터 (진품 + 가품 포함)
    print("\n1. 화가 분류용 데이터셋 (5클래스):")
    (X_train, y_train), (X_test, y_test) = load_impressionist_data(include_fake=True)
    
    print("\n2. 가품 검증용 데이터셋 (2클래스):")
    (X_train_forgery, y_train_forgery), (X_test_forgery, y_test_forgery) = load_forgery_detection_data()
    
    print("\n=" * 60)
    print("데이터 로드 완료!")
    print("=" * 60)
