import cv2
import torch

def run():
    print("Starting webcam...")

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access webcam")
        return
    cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Detection", 1200, 700)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame = results.render()[0]

        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# allow direct run too
if __name__ == "__main__":
    run()
