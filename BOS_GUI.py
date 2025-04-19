import cv2
import numpy as np
from scipy import fftpack, ndimage
import tkinter as tk
from tkinter import filedialog
import os

# --------------- helper functions -----------------

def load_gray(path):
    """load image in grayscale as float32"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot load {path}")
    return img.astype(np.float32)

def highpass(img, cutoff):
    """apply FFT-based high-pass filtering to remove low-frequency illumination"""
    f = fftpack.fft2(img)
    fshift = fftpack.fftshift(f)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    # create circular mask; frequencies below cutoff (scaled by image size) are suppressed
    mask = ((Y - cy)**2 + (X - cx)**2) > (cutoff * min(h, w))**2
    fshift *= mask
    return np.real(fftpack.ifft2(fftpack.ifftshift(fshift)))

def gabor_enhance(img, frequency, theta):
    """apply gabor filter tuned to the grid periodicity"""
    kernel = cv2.getGaborKernel(
        ksize=(31, 31),
        sigma=4.0,
        theta=theta,
        lambd=1.0 / frequency,
        gamma=0.5,
        psi=0
    )
    return cv2.filter2D(img, cv2.CV_32F, kernel)

def bos_process(bg, dist, blur_sigma, hp_cutoff, gabor_freq, theta_deg, gain):
    """process two grayscale images using BOS steps"""
    # gaussian blur to reduce high-frequency noise
    bg_f = ndimage.gaussian_filter(bg, sigma=blur_sigma)
    dist_f = ndimage.gaussian_filter(dist, sigma=blur_sigma)

    # high-pass filtering to remove illumination gradients
    bg_hp = highpass(bg_f, cutoff=hp_cutoff)
    dist_hp = highpass(dist_f, cutoff=hp_cutoff)

    # subtract and amplify differences
    diff = (dist_hp - bg_hp) * gain

    # apply gabor filter (convert theta to radians)
    theta = np.deg2rad(theta_deg)
    diff = gabor_enhance(diff, frequency=gabor_freq, theta=theta)

    # normalize to 0-255 and apply colormap for better visualization
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

def adjust_to_screen(img, screen_w, screen_h):
    """
    adjust image size to fit within the screen dimensions
    preserving aspect ratio.
    """
    h, w = img.shape[:2]
    scale = min(screen_w / w, screen_h / h, 1)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def nothing(x):
    pass

# --------------- GUI for image mode -----------------

def interactive_bos_images(ref_path, dist_path, screen_w, screen_h):
    """interactive bos gui for two image files (reference and disturbed)"""
    bg = load_gray(ref_path)
    dist = load_gray(dist_path)
    window_name = "BOS GUI - Images"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # fine-resolution sliders
    # blur sigma: 0.1 to 10.0 (scaled x10), default 1.0 → slider = 10
    cv2.createTrackbar("blur σ x10", window_name, 10, 100, nothing)
    # hp cutoff: 0.01 to 1.0 (scaled x100), default 0.02 → slider = 2
    cv2.createTrackbar("hp cutoff x100", window_name, 2, 100, nothing)
    # gabor frequency: 0.01 to 1.0 (scaled x100), default 0.25 → slider = 25
    cv2.createTrackbar("gabor freq x100", window_name, 25, 100, nothing)
    # theta: 0° to 180°, default 90
    cv2.createTrackbar("theta°", window_name, 90, 180, nothing)
    # gain: 0.1 to 50.0 (scaled x10), default 10.0 → slider = 100
    cv2.createTrackbar("gain x10", window_name, 100, 500, nothing)

    while True:
        # read parameter values from trackbars
        blur_sigma = cv2.getTrackbarPos("blur σ x10", window_name) / 10.0
        hp_cutoff  = cv2.getTrackbarPos("hp cutoff x100", window_name) / 100.0
        gabor_freq = cv2.getTrackbarPos("gabor freq x100", window_name) / 100.0
        theta_deg  = cv2.getTrackbarPos("theta°", window_name)
        gain       = cv2.getTrackbarPos("gain x10", window_name) / 10.0

        # enforce minimum values
        blur_sigma = max(0.1, blur_sigma)
        hp_cutoff  = max(0.001, hp_cutoff)
        gabor_freq = max(0.001, gabor_freq)
        gain       = max(0.1, gain)

        result = bos_process(bg, dist, blur_sigma, hp_cutoff, gabor_freq, theta_deg, gain)
        # adjust result image to fit screen
        display_img = adjust_to_screen(result, screen_w, screen_h)
        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(50)
        if key == 27:  # esc to exit
            break
    cv2.destroyAllWindows()

# --------------- GUI for video mode -----------------

def interactive_bos_video(video_path, screen_w, screen_h):
    """interactive bos gui for a video file with play button.
       user selects reference frame; disturbed frame can be set manually or auto played.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_name = "BOS GUI - Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # trackbars for frame selection
    cv2.createTrackbar("Ref Frame", window_name, 0, total_frames - 1, nothing)
    cv2.createTrackbar("Dist Frame", window_name, 1, total_frames - 1, nothing)
    # add a play toggle trackbar: 0 = pause, 1 = play
    cv2.createTrackbar("Play", window_name, 0, 1, nothing)

    # parameter sliders (same as images mode)
    cv2.createTrackbar("blur σ x10", window_name, 10, 100, nothing)
    cv2.createTrackbar("hp cutoff x100", window_name, 2, 100, nothing)
    cv2.createTrackbar("gabor freq x100", window_name, 25, 100, nothing)
    cv2.createTrackbar("theta°", window_name, 90, 180, nothing)
    cv2.createTrackbar("gain x10", window_name, 100, 500, nothing)

    auto_frame = 0  # for auto-playing the disturbed frame

    def get_frame(frame_no):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            return None
        # convert to grayscale
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    while True:
        ref_frame_no = cv2.getTrackbarPos("Ref Frame", window_name)
        # check if play mode is enabled
        play_mode = cv2.getTrackbarPos("Play", window_name)
        if play_mode == 1:
            dist_frame_no = auto_frame
            auto_frame = (auto_frame + 1) % total_frames
            # update the trackbar to reflect the current auto-play frame
            cv2.setTrackbarPos("Dist Frame", window_name, dist_frame_no)
        else:
            # if not playing, use the trackbar value
            dist_frame_no = cv2.getTrackbarPos("Dist Frame", window_name)

        # get processing parameters
        blur_sigma = cv2.getTrackbarPos("blur σ x10", window_name) / 10.0
        hp_cutoff  = cv2.getTrackbarPos("hp cutoff x100", window_name) / 100.0
        gabor_freq = cv2.getTrackbarPos("gabor freq x100", window_name) / 100.0
        theta_deg  = cv2.getTrackbarPos("theta°", window_name)
        gain       = cv2.getTrackbarPos("gain x10", window_name) / 10.0

        # safety clamps
        blur_sigma = max(0.1, blur_sigma)
        hp_cutoff  = max(0.001, hp_cutoff)
        gabor_freq = max(0.001, gabor_freq)
        gain       = max(0.1, gain)

        bg = get_frame(ref_frame_no)
        dist = get_frame(dist_frame_no)
        if bg is None or dist is None:
            print("error reading frames; check frame numbers or file")
            break

        result = bos_process(bg, dist, blur_sigma, hp_cutoff, gabor_freq, theta_deg, gain)
        # adjust result image to screen
        display_img = adjust_to_screen(result, screen_w, screen_h)
        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # esc to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------- main function with file browser -----------------

def main():
    # initialize tkinter (hidden root) and get screen resolution
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    print(f"screen resolution: {screen_w}x{screen_h}")

    filepaths = filedialog.askopenfilenames(
        title="Select 2 images (ref and disturbed) or 1 video file",
        filetypes=(
            ("all files", "*.*"),
            ("image files", "*.tif;*.png;*.jpg"),
            ("video files", "*.mp4;*.avi;*.mov;*.mkv")
        )
    )
    if not filepaths:
        print("no file selected, exiting")
        return

    if len(filepaths) == 2:
        ref_path, dist_path = filepaths
        print(f"loaded images:\n  reference: {ref_path}\n  disturbed: {dist_path}")
        interactive_bos_images(ref_path, dist_path, screen_w, screen_h)
    elif len(filepaths) == 1:
        video_path = filepaths[0]
        ext = os.path.splitext(video_path)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            print(f"loaded video: {video_path}")
            interactive_bos_video(video_path, screen_w, screen_h)
        else:
            print("selected single file is not a video; expecting 2 images for image mode.")
    else:
        print("please select either 2 files for images or 1 file for video mode.")

if __name__ == "__main__":
    main()
