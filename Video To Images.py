import cv2
import os

def video_to_images(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as PNG to avoid compression loss
        frame_filename = os.path.join(output_folder, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Done. Extracted {frame_idx} frames to '{output_folder}'.")

# Example usage
video_to_images("27.03.2025 BOS test03_1MHZ_13V_Video_short_pulses.avi", "27.03.2025 BOS test03_1MHZ_13V_Video_short_pulses Images")
