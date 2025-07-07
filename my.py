import argparse
import cv2
import torch
from pathlib import Path
import sys
import numpy as np

# Add yolov5 repo to path
sys.path.append(str(Path(__file__).parent / "yolov5"))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

from mysort import create_grape_tracker

def load_model(weights_path, device):
    model = DetectMultiBackend(weights_path, device=device)
    return model

def preprocess_frame(frame, img_size, stride):
    img = letterbox(frame, img_size, stride=stride)[0]
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = torch.from_numpy(img).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def draw_boxes(frame, detections, names):
    for *xyxy, conf, cls in detections:
        label = f'{names[int(cls)]} {conf:.2f}'
        xyxy = [int(x.item()) for x in xyxy]
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

def process_image(model, device, img_path, output_dir, img_size, stride, names):
    img0 = cv2.imread(img_path)
    img = preprocess_frame(img0, img_size, stride).to(device)
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()

    annotated = draw_boxes(img0, pred, names) if pred is not None else img0
    out_path = Path(output_dir) / f"det_{Path(img_path).name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved {out_path}")

def process_video(model, device, video_path, output_dir, img_size, stride, names):
    #tracker = Sort(max_age=10)
    tracker = create_grape_tracker()
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = Path(output_dir) / f"det_{Path(video_path).name}"
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    track_ids = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = preprocess_frame(frame, img_size, stride).to(device)
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)[0]
        
        #after NMS, feed detections into tracker
        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
            detections = []
            for *xyxy, conf, cls in pred:
                detections.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf)])
            detections = np.array(detections, dtype=np.float32)
        else:
            detections = np.empty((0, 5))

        trackers = tracker.update(detections)
        
        
        # Draw boxes with ID
        for d in trackers:
            x1, y1, x2, y2, track_id = d
            track_ids.append(track_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'ID {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        #annotated = draw_boxes(frame, pred, names) if pred is not None else frame
        print("count : ", len(set(track_ids)))
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Local Inference")
    parser.add_argument("input_path", help="Image or video path")
    parser.add_argument("--weights", default="yolov5s.pt", help="Path to YOLOv5 weights")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = select_device('')
    model = load_model(args.weights, device)
    names = model.names
    stride = model.stride

    if args.input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        process_image(model, device, args.input_path, args.output_dir, args.img_size, stride, names)
    elif args.input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(model, device, args.input_path, args.output_dir, args.img_size, stride, names)
    else:
        print("Unsupported file type")

if __name__ == "__main__":
    main()
