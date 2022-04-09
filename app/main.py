import cv2

from app.utils.config import load_yolov5_model


def get_annotations(video_path: str = 'resources/video.mp4') -> list:
    vid_capture = cv2.VideoCapture(video_path)
    results = []

    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
        exit()

    # Read fps and frame count
    else:
        # Get frame rate information
        fps = vid_capture.get(cv2.CAP_PROP_FPS)
        print('Frames per second : ', fps, 'FPS')

        # Get frame count
        frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        print('Frame count : ', frame_count)

    # Read video frame by frame
    while (vid_capture.isOpened()):
        exists, frame = vid_capture.read()
        if exists == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.transpose((2, 0, 1))
            result = yolo_v5_model(frame)
            result.print()
            current_timestamp = vid_capture.get(cv2.CAP_PROP_POS_MSEC)
            results.append((result.pandas().xyxy[0], current_timestamp))
        else:
            break
    # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()
    return results


if __name__ == '__main__':
    yolo_v5_model = load_yolov5_model()
    annotations_and_timestamps = get_annotations(video_path='resources/aquarium_video.mp4')
