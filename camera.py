from ultralytics import YOLO
import cv2

def main():
    # Caminho do modelo
    model = YOLO("runs/detect/modelo_v2/weights/best.pt")

    cap = cv2.VideoCapture(0)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("Detecção YOLOv8", annotated_frame)

        # Pressionar q para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
