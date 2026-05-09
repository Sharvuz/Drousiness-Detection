import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import pygame
import os

# ==========================================
# KHỐI 1: CẤU HÌNH ÂM THANH BÁO ĐỘNG
# ==========================================
pygame.mixer.init()

current_dir = os.path.dirname(os.path.abspath(__file__))
alarm_path = os.path.join(current_dir, "..", "media", "alarm.mp3")

try:
    pygame.mixer.music.load(alarm_path)
    print("Da tai thanh cong file am thanh!")
except Exception as e:
    print(f"LOI: Khong tim thay file am thanh tai {alarm_path}")
    print("Vui long kiem tra lai thu muc 'media' va file 'alarm.mp3'.")


# ==========================================
# KHỐI 2: HÀM TÍNH TOÁN EAR
# ==========================================
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# ==========================================
# KHỐI 3: KHỞI TẠO MÔ HÌNH DLIB VÀ CAMERA
# ==========================================
print("Dang khoi dong Camera va tai Model...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

# ==========================================
# CẤU HÌNH NGƯỠNG BUỒN NGỦ
# ==========================================
EAR_THRESHOLD = 0.3
CLOSED_EYE_LIMIT = 50
closed_eye_counter = 0

# ==========================================
# KHỐI 4: VÒNG LẶP XỬ LÝ
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Dem mat nham: {closed_eye_counter}/{CLOSED_EYE_LIMIT}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Nếu EAR nhỏ hơn 0.3 thì tính là mắt đang nhắm
        if ear < EAR_THRESHOLD:
            closed_eye_counter += 1

            # Nếu mắt nhắm liên tục đủ 50 lần thì báo động
            if closed_eye_counter >= CLOSED_EYE_LIMIT:
                cv2.putText(
                    frame,
                    "BUON NGU!",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()

        else:
            # Nếu mắt mở lại thì reset bộ đếm và tắt báo động
            closed_eye_counter = 0
            pygame.mixer.music.stop()

    cv2.imshow("He Thong Canh Bao Buon Ngu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
