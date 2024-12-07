import cv2
import numpy as np

def preprocess_frame(frame, target_size=(1000, 600)):
    frame = cv2.resize(frame, target_size)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frame, hsv_frame

def detect_table(hsv_frame, min_table_area=50000):
    lower_green = np.array([30, 40, 40], dtype="uint8")
    upper_green = np.array([85, 255, 255], dtype="uint8")
    table_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    if cv2.countNonZero(table_mask) < min_table_area:
        return None
    return table_mask

def segment_color(hsv_frame, color_range, table_mask):
    lower = np.array(color_range[0], dtype="uint8")
    upper = np.array(color_range[1], dtype="uint8")
    mask = cv2.inRange(hsv_frame, lower, upper)

    if table_mask is not None:
        mask = cv2.bitwise_and(mask, mask, mask=table_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return refined_mask

def detect_balls(mask, min_radius=2, max_radius=10):
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 1)
    circles = cv2.HoughCircles(
        blurred_mask, cv2.HOUGH_GRADIENT, dp=1.5, minDist=15,
        param1=50, param2=20, minRadius=min_radius, maxRadius=max_radius
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return [(x, y, radius) for x, y, radius in circles[0]]
    return []

def annotate_balls(frame, balls, color_name):
    for (x, y, radius) in balls:
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        cv2.putText(frame, color_name, (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def snooker_ball_detection(video_source, ball_colors, selected_colors=None):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_skip = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame, hsv_frame = preprocess_frame(frame)
        table_mask = detect_table(hsv_frame)
        if table_mask is None:
            cv2.imshow("Snooker Ball Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # If no color is selected, process all colors
        colors_to_detect = selected_colors if selected_colors else ball_colors.keys()

        for color_name in colors_to_detect:
            color_range = ball_colors[color_name]
            color_mask = segment_color(hsv_frame, color_range, table_mask)
            balls = detect_balls(color_mask)
            annotate_balls(frame, balls, color_name)

        cv2.imshow("Snooker Ball Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Define HSV ranges for each ball color
ball_colors = {
    "White": ([0, 0, 200], [180, 30, 255]),
    "Yellow": ([25, 150, 150], [35, 255, 255]),
    # "Green": ([50, 100, 100], [70, 255, 255]),
    "Brown": ([10, 100, 100], [20, 255, 255]),
    "Black": ([0, 0, 0], [180, 255, 50]),
    "Blue": ([100, 150, 50], [140, 255, 255]),
    "Pink": ([150, 100, 100], [170, 255, 255]),
    "Red": ([0, 120, 70], [10, 255, 255]),
}

# Prompt user for video source
print("Enter '0' for webcam or the path to a video file:")
video_source = input().strip()

# Default to webcam if '0' is entered, else use the file path
video_source = 0 if video_source == "0" else video_source

# Allow user to select colors dynamically
print("Available colors:", ", ".join(ball_colors.keys()))
print("Enter colors to detect (comma-separated) or press Enter to detect all colors:")
selected_colors = input().strip().split(",")

# Remove whitespace and validate input
selected_colors = [color.strip() for color in selected_colors if color.strip() in ball_colors]

# Run detection
snooker_ball_detection(video_source, ball_colors, selected_colors)
