from ultralytics import YOLO
from video_utils import read_video, save_video
import cv2
import pickle

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        """This function returns a dictionary containing the key of each player and the value of bbox."""
        model = self.model
        tracker = model.track(frame, persist=True)[0]
        tracker_id = tracker.names  # dict
        player_dict = {}
        for box in tracker.boxes:
            # print(box)
            box_id = int(box.id.tolist()[0])
            xyxy = box.xyxy.tolist()[0]
            player_id = box.cls.tolist()[0]
            player_name = tracker_id[player_id]
            if player_name == "Player1":
                player_dict[box_id] = xyxy
            else:
                player_dict[box_id] = xyxy
            # print(player_dict)
        return player_dict

    def detect_player(self, frames, last_detect=False, path_of_last_detect=None):
        """This function detects the player in each frame and returns it as a list of dictionaries containing bbox."""
        # read last detect player
        if last_detect and path_of_last_detect is not None:
            with open(path_of_last_detect, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        player_detections = []
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if path_of_last_detect is not None:
            with open(path_of_last_detect, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def player_positions(frames, detections):
        c_positions = {}
        c_bboxes = []
        for k, bbox in zip(detections.keys(), detections.values()):
            x1,y1,x2,y2=det
            c_x = int(x2-x1)/2
            c_y = int(y2-y1)/2
            id = k
            c_bboxes.append(c_x)
            c_bboxes.append(c_y)
            c_positions = {id : c_bboxes}
        return c_positions        
    
    def draw_player_bbox(self, frames, player_detections):
        player_frames = []
        for frame, player_detect in zip(frames, player_detections):
            for id, box in player_detect.items():
                x1, y1, x2, y2 = box
                if id == 1:
                    cv2.putText(frame, f"Player: {id}", (int(box[0]), int(box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"Player: {id}", (int(box[0]), int(box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            player_frames.append(frame)
        return player_frames

if __name__ == '__main__':
    input_video_path = '/home/Avdesh/PycharmProjects/Badminton-Player-Tracking/test_video.mp4'
    
    frames = read_video(input_video_path)
    
    model_path = '/home/Avdesh/PycharmProjects/Badminton-Player-Tracking/train/player_output/models/weights/best.pt'
    PlayerTracker(model_path).detect_frame(frames[100])
    
