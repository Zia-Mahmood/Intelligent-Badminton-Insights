from ultralytics import YOLO
import cv2
from video_utils import read_video, save_video
import pickle
import pandas as pd

class ShuttleTracker:
    def __init__(self, model_path=None):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        """This function return list of dict"""
        model = self.model
        shuttle_track = model.track(frame, persist=True)[0]
        boxes = shuttle_track.boxes
        names = shuttle_track.names
        shuttle_dict = {}  # {name: [bbox]}
        for box in boxes:
            cls = int(box.cls.tolist()[0])
            xyxy = box.xyxy.tolist()[0]
            shuttle_name = names[cls]
            if shuttle_name:
                shuttle_dict[cls] = xyxy
                # print(shuttle_dict)
        return shuttle_dict

    def detect_shuttle(self, frames, last_detect=False, path_of_last_detect=None):
        """This function returns a dictionary containing the key of shuttle and the value of bbox."""
        # read last detect shuttle
        if last_detect and path_of_last_detect is not None:
            with open(path_of_last_detect, 'rb') as f:
                shuttle_detections = pickle.load(f)
            return shuttle_detections

        shuttle_detections = []
        for frame in frames:
            shuttle_dict = self.detect_frame(frame)
            shuttle_detections.append(shuttle_dict)

        if path_of_last_detect is not None:
            with open(path_of_last_detect, 'wb') as f:
                pickle.dump(shuttle_detections, f)

        return shuttle_detections

    def interpolate_shuttle_position(self,shuttle_positions):
        positions = [x.get(0, []) for x in shuttle_positions]
        # convert list into pandas dataframe
        df_shuttle_positions = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_shuttle_positions = df_shuttle_positions.interpolate()
        # No detect at first frame
        df_shuttle_positions = df_shuttle_positions.bfill()

        # convert back to list
        shuttle_positions = [{0: x} for x in df_shuttle_positions.to_numpy().tolist()]
        return shuttle_positions


    def draw_shuttle_bbox(self, frames, shuttle_detections):
        shuttle_frames = []
        for frame, shuttle_detect in zip(frames, shuttle_detections):
            for name, bbox in shuttle_detect.items():
                if name==0:
                    x1, y1, x2, y2 = bbox
                    # print(x1, y1, x2, y2)
                    cv2.putText(frame, "Shuttle: {}".format(name), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color=(0, 255, 0), thickness=2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            shuttle_frames.append(frame)
        return shuttle_frames

if __name__ == '__main__':
    input_video_path = '/home/Avdesh/PycharmProjects/Badminton-Player-Tracking-and-Analysis/test_video.mp4'
    # read video
    frames = read_video(input_video_path)

    model = '/home/Avdesh/PycharmProjects/Badminton-Player-Tracking-and-Analysis/train/shuttle_output/models/weights/best.pt'

    # save video
    output_video_path = 'output_video1.mp4'
    save_video(frames, input_video_path, output_video_path)

