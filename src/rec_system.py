import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# --- 1. DATA MAPPING (ĐỊNH NGHĨA KHÔNG GIAN CẢM XÚC) ---
# Ánh xạ Cảm xúc -> Tọa độ Vector [Valence, Energy] lý tưởng
EMOTION_TARGETS = {
    'happy': [0.85, 0.80],  # Vui vẻ: Tích cực cao, Năng lượng cao
    'sad': [0.20, 0.20],  # Buồn: Tích cực thấp, Năng lượng thấp
    'neutral': [0.50, 0.50],  # Bình thường: Cân bằng
    'drowsy': [0.80, 0.90],  # Buồn ngủ: Cần nhạc cực mạnh và vui để kích thích tỉnh táo
    'angry': [0.15, 0.85],  # Tức giận: Tích cực thấp, Năng lượng rất cao
    'surprised': [0.70, 0.80],  # Bất ngờ: Tích cực khá, Năng lượng cao
    'scared': [0.30, 0.70],  # Sợ hãi: Tích cực thấp, Năng lượng khá cao
    'disgust': [0.25, 0.60]  # Ghê tởm: Tích cực thấp, Năng lượng vừa
}


# --- 2. TRAIN MODEL (HUẤN LUYỆN) ---
def train_knn_model(df):
    """
    Huấn luyện mô hình KNN trên đặc trưng của các bài hát.
    """
    # Trích xuất 2 cột đặc trưng (Features) thành ma trận NumPy
    X = df[['valence', 'energy']].values

    # Khởi tạo mô hình KNN. Thuật toán 'brute' tính toán khoảng cách Euclidean
    # n_neighbors=15 nghĩa là ta luôn lấy ra 15 bài hát gần nhất
    knn = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean')

    # Cho mô hình "học" vị trí của tất cả bài hát
    knn.fit(X)

    return knn


# --- 3. INFERENCE (DỰ ĐOÁN & GỢI Ý) ---
def recommend_ml(df, knn_model, emotion):
    """
    Nhận cảm xúc, biến thành vector và tìm bài hát gần nhất.
    """
    # Bước 1: Lấy tọa độ mục tiêu. Nếu AI đọc ra cảm xúc lạ, mặc định cho về Neutral
    target_vector = EMOTION_TARGETS.get(emotion, [0.5, 0.5])

    # Bước 2: Đưa về mảng 2D cho Scikit-learn (VD: [[0.2, 0.2]])
    target_array = np.array([target_vector])

    # Bước 3: Đo khoảng cách và lấy index của các bài hát gần nhất
    distances, indices = knn_model.kneighbors(target_array)

    # Bước 4: Trích xuất các bài hát từ DataFrame gốc dựa trên index
    recommended_songs = df.iloc[indices[0]]

    return recommended_songs
