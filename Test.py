from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import cv2
import time

def interpolate(model, frame, path):
    tracker = model.predictor.trackers[0]
    tracks = [t for t in tracker.tracked_stracks if t.is_activated]

    # Ensure predictions are made for active tracks
    if tracks:
        tracker.multi_predict(tracks)
        tracker.frame_id += 1
        boxes = [np.hstack([t.xyxy, t.track_id, t.score, t.cls]) for t in tracks]
        
        # Update frame IDs for all active tracks
        for t in tracks:
            t.frame_id = tracker.frame_id

        return Results(frame, path, model.names, np.array(boxes))
    else:
        # No active tracks; return an empty result
        return Results(frame, path, model.names, np.empty((0, 6)))


def infer_on_video(model, pose, filename, output, stride, start_frame=5):
    cap = cv2.VideoCapture(filename)

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_id = 1
    path = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (0, 255, 0)
    thickness = 2
    bg_color = (0, 0, 0)
    padding = 5

    # Counters for masks
    mask_count = 0
    no_mask_count = 0

    # Track IDs to avoid double counting
    counted_ids = set()

    start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % stride != 0 and frame_id >= start_frame:
            result = interpolate(model, frame, path)
        else:
            result = model.track(frame, persist=True, verbose=False)[0]
            if path is None:
                path = result.path

        # Counting logic
        # Counting logic
        for box in result.boxes:
            if box.id is not None:  # Only process valid boxes
                track_id = int(box.id)
                cls = int(box.cls)

                if track_id not in counted_ids:
                    if cls == 0:  # Assuming 'mask' is class 0
                        mask_count += 1
                    elif cls == 1:  # Assuming 'no mask' is class 1
                        no_mask_count += 1
                    counted_ids.add(track_id)

        pose_results = pose.track(frame, persist=True, verbose=False)[0]
        for box in pose_results.boxes:
            cls = int(box.cls)
            if cls == 0: 
                # Highlight falling person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "FALL", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        annotated = result.plot()
        avg_fps = frame_id / (time.perf_counter() - start)
        fps_text = f"FPS (stride={stride}): {avg_fps:.2f}"
        count_text = f"Masks: {mask_count}, No Masks: {no_mask_count}"

        text_size = cv2.getTextSize(fps_text, font, font_scale, thickness)[0]
        count_size = cv2.getTextSize(count_text, font, font_scale, thickness)[0]

        text_x, text_y = 10, 30
        count_x, count_y = 10, text_y + 40

        cv2.rectangle(annotated, (text_x - padding, text_y - text_size[1] - padding),
                      (text_x + text_size[0] + padding, text_y + padding), bg_color, -1)
        cv2.rectangle(annotated, (count_x - padding, count_y - count_size[1] - padding),
                      (count_x + count_size[0] + padding, count_y + padding), bg_color, -1)

        cv2.putText(annotated, fps_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(annotated, count_text, (count_x, count_y), font, font_scale, color, thickness, cv2.LINE_AA)

        writer.write(annotated)
        frame_id += 1

    writer.release()
    cap.release()
    print(f"Final Counts - Masks: {mask_count}, No Masks: {no_mask_count}")

# Reset trackers
model = YOLO("masknew.pt").to("cuda")
model(verbose=False)
pose = YOLO("fall.pt").to("cuda")
infer_on_video(model, pose, "mask.mp4", "cuda-server.mp4", stride=3)
