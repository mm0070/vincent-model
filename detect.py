import torch
import cv2

"""
The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
"""


def score_frame(frame, model):
    model.to("cpu")
    frame = [torch.tensor(frame)]
    results = self.model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    cord = results.xyxyn[0][:, :-1].numpy()
    return labels, cord


"""
The function below takes the results and the frame as input and plots boxes over all the objects which have a score higer than our threshold.
"""


def plot_boxes(self, results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2:
            continue
        x1 = int(row[0] * x_shape)
        y1 = int(row[1] * y_shape)
        x2 = int(row[2] * x_shape)
        y2 = int(row[3] * y_shape)
        bgr = (0, 255, 0)  # color of the box
        classes = self.model.names  # Get the name of label index
        label_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the label.
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)  # Plot the boxes
        cv2.putText(
            frame, classes[labels[i]], (x1, y1), label_font, 0.9, bgr, 2
        )  # Put a label over box.
        return frame


def main():
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path="vincent-epochs-20-batch-16/weights/best.pt",
    )

    camera_ip = "rtsp://vincent:goodboy@192.168.1.229/stream1"
    # player = cv2.VideoCapture(camera_ip)  # tapo
    # player = cv2.VideoCapture(0)  # macbook
    player = cv2.VideoCapture("vincent_compilation.mp4")  # file
    assert player.isOpened()


if __name__ == "__main__":
    main()
