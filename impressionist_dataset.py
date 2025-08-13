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
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
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

def load_impressionist_data(split=True, test_size=0.2, target_size=(224, 224), local_path=None):
    """
    인상파 화가 데이터셋을 로드합니다.
    
    Parameters:
    -----------
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
    """
    
    # 화가별 라벨 매핑
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
    
    print("이미지를 로드하는 중...")
    
    # 각 화가별 이미지 로드
    for artist_name, label in artist_labels.items():
        folder_path = os.path.join(data_dir, artist_name)
        images, labels = load_images_from_folder(folder_path, label, target_size)
        
        if len(images) > 0:
            all_images.extend(images)
            all_labels.extend(labels)
            print(f"{artist_name}: {len(images)}개 이미지 로드 완료")
        else:
            print(f"{artist_name}: 이미지를 찾을 수 없습니다")
    
    # numpy 배열로 변환
    X = np.array(all_images)
    y = np.array(all_labels)
    
    print(f"총 {len(X)}개 이미지 로드 완료")
    print(f"이미지 shape: {X.shape}")
    print(f"라벨 분포: {np.bincount(y)}")
    
    # 데이터 정규화 (0-1 범위)
    X = X.astype('float32') / 255.0
    
    if split:
        # train/test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]}개")
        print(f"Test set: {X_test.shape[0]}개")
        
        return (X_train, y_train), (X_test, y_test)
    else:
        return X, y

def get_class_names():
    """클래스 이름을 반환합니다."""
    return ['Alfred_Sisley', 'Camille_Pissarro', 'Claude_Monet', 'Pierre-Auguste_Renoir']

# 사용 예제
if __name__ == "__main__":
    # MNIST 스타일로 데이터 로드
    (X_train, y_train), (X_test, y_test) = load_impressionist_data()
    
    print("데이터셋 정보:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"클래스 수: {len(get_class_names())}")
    print(f"클래스 이름: {get_class_names()}")
