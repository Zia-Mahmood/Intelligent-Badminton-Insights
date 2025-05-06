import cv2
from video_utils import read_video, save_video
from shuttle_tracker import ShuttleTracker


def main():
    ### input video ###
    input_video_path = 'test_video.mp4'
    # read video
    frames = read_video(input_video_path)

    # shuttle detection
    shuttle_tracker = ShuttleTracker(model_path="train/shuttle_output/models/weights/best.pt", )
    shuttle_detect = shuttle_tracker.detect_shuttle(frames, last_detect=True, path_of_last_detect="last_detect/list_shuttle_dict.pkl")
    shuttle_interpolate = shuttle_tracker.interpolate_shuttle_position(shuttle_detect)

    ### draw ###
    # draw shuttle bbox
    output_frames = shuttle_tracker.draw_shuttle_bbox(output_frames, shuttle_interpolate)

    ### output video ###
    # save video
    output_video_path = 'output_video.mp4'
    save_video(output_frames, input_video_path, output_video_path)


if __name__ == '__main__':
    main()
