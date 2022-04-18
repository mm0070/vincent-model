import torch
import cv2
from time import time
import numpy as np


def score_frame(model, device, frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -
                                    1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
    return labels, cord


def class_to_label(model, x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    classes = model.names
    return classes[int(x)]


def plot_boxes(results, frame, model, min_confidence=0.05):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= min_confidence:
            x1, y1, x2, y2 = (
                int(row[0] * x_shape),
                int(row[1] * y_shape),
                int(row[2] * x_shape),
                int(row[3] * y_shape),
            )
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(
                frame,
                class_to_label(model, labels[i]) + " " + str(format(row[4], '.2f')),
                (x1, y1),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                bgr,
                2,
            )

    return frame

def main():
    # OpenCV capture setup
    camera_ip = "rtsp://vincent:goodboy@192.168.1.229/stream1"
    # player = cv2.VideoCapture(camera_ip)  # Tapo
    # player = cv2.VideoCapture(0)  # MacBook webcam
    player = cv2.VideoCapture("vincent_compilation.mp4")  # File
    assert player.isOpened()

    # Get image size details
    x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
    four_cc = cv2.VideoWriter_fourcc(*"MJPG")

    # Load my magnificent model
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path="vincent-e20-b16-v5l.pt",
    )

    # Cross fingers and hope cuda is working
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Running on CPU, CUDA is broken!")

    # Window display
    while player.isOpened():
        # Capture frame-by-frame
        ret, frame = player.read()
        if ret == True:
            start_time = time()
            results = score_frame(model, device, frame)
            frame = plot_boxes(results, frame, model)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")

            # Display the resulting frame
            cv2.imshow("Frame", frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        # Break the loop
        else:
            break

    # When everything is done, release the video capture object
    player.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
