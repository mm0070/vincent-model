from nbformat import write
import torch
import cv2
from time import time
import numpy as np
import json

CAMERA_IP = "rtsp://vincent:goodboy@192.168.1.229/stream1"
MIN_CONFIDENCE = 0.05

def score_frame(model, device, frame):
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
    return labels, cord


def class_to_label(model, x):
    classes = model.names
    return classes[int(x)]

def plot_boxes(confidence, coords, frame):
    # plot box over predicted Vincent
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    x1, y1, x2, y2 = (
                int(coords[0] * x_shape),
                int(coords[1] * y_shape),
                int(coords[2] * x_shape),
                int(coords[3] * y_shape),
            )
    bgr = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
    cv2.putText(
        frame,
        "VincentVanDogh" +
        " " + format(confidence, '.2f'),
        (x1, y1),
        cv2.FONT_HERSHEY_COMPLEX,
        0.9,
        bgr,
        2,
    )
    return frame

def get_best_prediction(results, model, min_confidence=0.05):
    # returns confidence and coordinates of the best prediction of Vincent
    labels, cord = results
    n = len(labels)
    best = np.array([0, 0])
    for i in range(n):
        row = cord[i]
        confidence = row[4]
        if confidence >= min_confidence:
            name = class_to_label(model, labels[i])
            if name == 'vincent':
                if confidence > best[1]:
                    best = [i, confidence]

    confidence = best[1]
    coords = np.array([0,0,0,0])

    # get coords of the best guess
    if len(cord) > 0:
        row = cord[best[0]]
        coords = np.array([row[0], row[1], row[2], row[3]]) #x1, y1, x2, y2

    return confidence, coords



def write_json(center_points_x, center_points_y, x_shape, y_shape, confidence_ot, filename='detect.json'):
    data = {
        "average_position_x": np.average(center_points_x),
        "average_position_y": np.average(center_points_y),
        "average_position_delta_x": np.abs(np.average(np.ediff1d(center_points_x))),
        "average_position_delta_y": np.abs(np.average(np.ediff1d(center_points_y))),
        "canvas_size_x": x_shape,
        "canvas_size_y": y_shape,
        "average_confidence": np.average(confidence_ot)
    }
    with open(filename, 'w') as fp:
        json.dump(data, fp,  indent=4)



def main():
    # OpenCV capture setup
    # player = cv2.VideoCapture(CAMERA_IP)  # Tapo
    # player = cv2.VideoCapture(0)  # MacBook webcam
    player = cv2.VideoCapture("vincent_compilation.mp4")  # File
    assert player.isOpened()

    # Get image size details
    x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # Initialize frame counter
    frame_counter = 0

    center_points_x = np.array([])
    center_points_y = np.array([])
    confidence_ot = np.array([])

    # Window display
    while player.isOpened():
        # Capture frame-by-frame
        ret, frame = player.read()
        if ret == True:
            # Invoke the AI and hope it doesn't become Skynet
            start_time = time()
            results = score_frame(model, device, frame)
            confidence, coords = get_best_prediction(results, model)
            frame = plot_boxes(confidence, coords, frame)

            # Hold results in array to calculate averages
            if confidence >= MIN_CONFIDENCE:
                center_x = ((coords[0] + coords[2]) / 2) * x_shape # (x1+x2)/2 * feed X - find X midpoint in pixels
                center_y = ((coords[1] + coords[3]) / 2) * y_shape # (y1+y2)/2 * feed Y - find Y midpoint in pixels
                center_points_x = np.append(center_points_x, center_x)
                center_points_y = np.append(center_points_y, center_y)
                confidence_ot = np.append(confidence_ot, confidence)

            frame_counter += 1
            if frame_counter >= 25:  # 25FPS, run every 10s
                # Reset the counter to 0
                frame_counter = 0

                # Write to json
                write_json(center_points_x, center_points_y, x_shape, y_shape, confidence_ot)

                # Zero the center point and confidence arrays
                center_points_x = np.array([])
                center_points_y = np.array([])
                confidence_ot = np.array([])

            # Log stuff to console
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            print(f"Confidence: {confidence}")

            # Display the resulting frame
            cv2.imshow("Frame", frame)

            # Press Q to exit
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
