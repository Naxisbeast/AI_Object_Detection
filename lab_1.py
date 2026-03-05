import argparse
import sys
from pathlib import Path

from ultralytics import YOLO
import cv2

# ─────────────────────────────────────────────
#  Lab 1 – YOLOv8 Object Detection & Tracking
# ─────────────────────────────────────────────

DEFAULT_MODEL  = "yolov8n.pt"
DEFAULT_IMAGE  = "data/test_image.jpg"
DEFAULT_VIDEO  = "data/test_video.mp4"
IMAGE_OUTPUT   = "results/image_detection_output.jpg"
VIDEO_OUTPUT   = "results/video_tracking_output.mp4"


def _require_file(path: str) -> None:
    """Exit with a helpful message when a required input file is missing."""
    if not Path(path).is_file():
        print(f"ERROR: Input file not found: '{path}'")
        print("  Make sure you have placed the file at the expected path before running.")
        sys.exit(1)


def detect_image(model: YOLO, image_path: str) -> None:
    """Run object detection on a single image and save the annotated result."""
    _require_file(image_path)
    print("\n[1/3] Running object detection on image...")
    results = model.predict(source=image_path, save=False, conf=0.25)

    # Render and save the first result frame
    annotated = results[0].plot()
    cv2.imwrite(IMAGE_OUTPUT, annotated)
    print(f"    Detection complete. Output saved to: {IMAGE_OUTPUT}")


def track_video(model: YOLO, video_path: str) -> None:
    """Run object tracking on a video file and save the annotated output."""
    _require_file(video_path)
    print("\n[2/3] Running object tracking on video...")
    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if not raw_fps or raw_fps <= 0:
        print("    WARNING: Could not read FPS from video; defaulting to 30 fps.")
        fps = 30.0
    else:
        fps = raw_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results   = model.track(source=frame, persist=True, conf=0.25, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()
    print(f"    Tracking complete ({frame_count} frames). Output saved to: {VIDEO_OUTPUT}")


def detect_webcam(model: YOLO) -> None:
    """Run real-time object detection using the default webcam.

    Press 'q' to quit.
    """
    print("\n[3/3] Starting real-time webcam detection (press 'q' to quit)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("    WARNING: No webcam detected. Skipping webcam demo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results   = model.predict(source=frame, conf=0.25, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("YOLOv8 – Real-Time Detection (press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("    Webcam session ended.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 1 – YOLOv8 Object Detection & Tracking"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to YOLOv8 weights file (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Path to input image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help=f"Path to input video (default: {DEFAULT_VIDEO})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Lab 1 – YOLOv8 Object Detection & Tracking")
    print("=" * 60)

    model = YOLO(args.model)

    detect_image(model, args.image)
    track_video(model, args.video)
    detect_webcam(model)

    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()
