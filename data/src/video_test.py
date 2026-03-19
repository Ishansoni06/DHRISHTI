import cv2

video_path = "data/test.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

frame_interval = int(fps)  # 1 second

frame_id = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        filename = f"frame_{saved_count}.jpg"
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_id += 1

cap.release()

print("Saved frames:", saved_count)