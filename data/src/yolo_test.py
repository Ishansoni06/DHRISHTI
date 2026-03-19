import cv2
from ultralytics import YOLO
import torch
import json
from itertools import groupby
# Load YOLO model
model = YOLO("yolov8n.pt")  # make sure this file is downloaded
model.to("cuda:0")
# print(next(model.model.parameters()).device)
# Load video
cap = cv2.VideoCapture("data/video-test.mp4")

#create events.
fps = cap.get(cv2.CAP_PROP_FPS)   
print(fps)
frame_id = 0                     
events = [] 


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on frame
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        #ignore low confidence
        if confidence < 0.5:
            continue

        # Draw box for ALL objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show label instead of "Person"
        cv2.putText(frame, f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        # Store event
        if label in ['toothbrush','person']:
            events.append({
                "frame": frame_id,
                "time": frame_id / fps,
                "object": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })
    
    # Show video
    cv2.imshow("Detection", frame)
    cv2.setWindowProperty("Detection", cv2.WND_PROP_TOPMOST, 1)
    # hwnd = win32gui.FindWindow(None, "Detection")
    # win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    # win32gui.SetForegroundWindow(hwnd)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_id += 1

cap.release()
cv2.destroyAllWindows()
#dumping events.json
with open("data/events.json", "w") as f:
    json.dump(events, f, indent=4)

#grouping of events

GAP_TOLERANCE = 10  # frames missing before considered a new event

# Sort by object type, then frame
events.sort(key=lambda e: (e["object"], e["frame"]))

grouped_events = []

# Group by object type first
for obj_type, obj_events in groupby(events, key=lambda e: e["object"]):
    obj_events = list(obj_events)

    # Start the first group
    current_group = {
        "object": obj_type,
        "start_frame": obj_events[0]["frame"],
        "end_frame": obj_events[0]["frame"],
        "detections": [obj_events[0]],
    }

    for event in obj_events[1:]:
        gap = event["frame"] - current_group["end_frame"]

        if gap <= GAP_TOLERANCE:
            # Gap is small — model likely missed it, extend the group
            current_group["end_frame"] = event["frame"]
            current_group["detections"].append(event)
        else:
            # Gap too large — genuine disappearance, save and start new group
            grouped_events.append(current_group)
            current_group = {
                "object": obj_type,
                "start_frame": event["frame"],
                "end_frame": event["frame"],
                "detections": [event],
            }

    grouped_events.append(current_group)  # don't forget the last group

# Clean up: compute summary fields, drop raw detections if you want
output = []
for group in grouped_events:
    detections = group["detections"]
    avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
    output.append({
        "object": group["object"],
        "start_frame": group["start_frame"],
        "end_frame": group["end_frame"],
        "duration_frames": group["end_frame"] - group["start_frame"] + 1,
        "detection_count": len(detections),
        "avg_confidence": round(avg_confidence, 4),
        # Optionally keep the raw detections:
        # "detections": detections,
    })
with open("data/grouped_events.json", "w") as f:
    json.dump(output, f, indent=4)

