import torch
import numpy as np
import cv2
from time import time


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, url, out_file="Labeled_Video.avi"):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = "cpu"

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="vincent-epochs-20-batch-16/weights/best.pt",
        )
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
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
            if row[4] >= 0.05:
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
                    self.class_to_label(labels[i]) + " " + str(row[4]),
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bgr,
                    2,
                )

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        # camera_ip = "rtsp://vincent:goodboy@192.168.1.229/stream1"
        camera_ip = "rtsp://localhost:1234/stream1"
        # player = cv2.VideoCapture(camera_ip)  # tapo
        # player = cv2.VideoCapture(0)  # macbook
        player = cv2.VideoCapture("vincent_compilation.mp4")  # file
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))

        counter = 25
        # Window display
        while player.isOpened():
            # Capture frame-by-frame
            ret, frame = player.read()
            if ret == True:
                start_time = time()
                if counter >= 25:
                    results = self.score_frame(frame)
                    frame = self.plot_boxes(results, frame)
                    counter = 0
                counter += 1
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 3)
                print(f"Frames Per Second : {fps}")
                out.write(frame)
                # Display the resulting frame
                cv2.imshow("Frame", frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        player.release()

        # Closes all the frames
        cv2.destroyAllWindows()


# Create a new object and execute.
a = ObjectDetection("https://www.youtube.com/watch?v=dwD1n7N7EAg")
a()
