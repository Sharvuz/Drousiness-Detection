import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


# --- 1. HÀM XÂY DỰNG KIẾN TRÚC MÔ HÌNH ---
def emotion_recognition(input_shape=(48, 48, 1)):
    """
    Hàm này tạo ra kiến trúc của mạng CNN để nhận diện 7 loại cảm xúc.
    - input_shape: Kích thước ảnh đầu vào. Ảnh xám (grayscale) 48x48 nên có shape là (48, 48, 1).
    """
    model = Sequential()

    # Block 1: Trích xuất đặc trưng cơ bản (Cạnh, góc...)
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Thu nhỏ kích thước ảnh
    model.add(Dropout(0.25))  # Tránh học vẹt (Overfitting)

    # Block 2: Trích xuất đặc trưng phức tạp hơn
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3: Chuyển đổi thành mảng 1 chiều và Phân loại
    model.add(Flatten())  # Kéo giãn ma trận ảnh thành 1 vector
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # Layer cuối: Xuất ra 7 xác suất cho 7 cảm xúc (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
    model.add(Dense(7, activation='softmax'))

    return model


# --- 2. HÀM HUẤN LUYỆN MÔ HÌNH ---
def trainModel(num_epochs=30, batch_size=64):
    """
    Hàm này dùng để huấn luyện mô hình nếu em chọn chạy lệnh: python emotion_detection.py --mode train
    LƯU Ý: Để hàm này chạy được, em cần có thư mục dữ liệu 'data/train' và 'data/validation' chứa ảnh.
    """
    print("Đang chuẩn bị huấn luyện mô hình...")

    # Khởi tạo mô hình
    model = emotion_recognition((48, 48, 1))

    # Biên dịch mô hình: dùng Categorical Crossentropy vì đây là bài toán phân loại nhiều lớp
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Tiền xử lý dữ liệu (Data Augmentation)
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load data từ thư mục (Giả định em cấu
