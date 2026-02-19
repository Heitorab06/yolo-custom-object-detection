import cv2
import time
import os

os.makedirs("images", exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
interval = 0.5  # segundos

last_capture = time.time()

while True:
    ret, frame = cap.read()
    cv2.imshow("Captura automática", frame)

    current_time = time.time()

    if current_time - last_capture >= interval:
        cv2.imwrite(f"images/celular2_{count}.jpg", frame)
        print(f"Imagem {count} salva")
        count += 1
        last_capture = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
