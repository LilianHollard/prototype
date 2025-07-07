import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import multiprocessing as mp

# Add yolov5 repo to path
sys.path.append(str(Path(__file__).parent / "yolov5"))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from bisort import create_grape_tracker

video_output = False


#Load model, using MultiBackend function from YOLOv5 

#Pour l'équipe : DetectMultiBackend fonctionne très bien sur YOLOv5 + pas de surcouche de code très haut niveau comme ultralytics
#                Globalement, il faut utiliser le programme "export.py", et exporter notre modèle de grappe directement vers TensorRT.
#                La fonction "load model" se chargera de faire la lecture d'un poids sous tensorrt sans pb.
def load_model(weights_path, device):
    model = DetectMultiBackend(weights_path, device=device)
    return model

#Format OpenCV --> to YOLO
def preprocess_frame(frame, img_size, stride):
    img = letterbox(frame, img_size, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW
    img = np.ascontiguousarray(img) 
    img = torch.from_numpy(img).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


#Thread 1 : detection
#Le tracker peut parfois prendre du temps qu'on ne veut pas gaspiller ! 
#Il y a une file de données : det_queue
#Ce thread récupère une image, la traite, puis envoie les bbox dans la file.
def detector_worker(video_path, img_size, stride, names, weights, device_str, det_queue):
    device = select_device(device_str)
    model = load_model(weights, device)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            det_queue.put(None)
            break
        img = preprocess_frame(frame, img_size, stride).to(device)
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.2, 0.45, agnostic=False)[0]
        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
            detections = []
            for *xyxy, conf, cls in pred:
                detections.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf)])
                if video_output:
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 4)
            detections = np.array(detections, dtype=np.float32)
        else:
            detections = np.empty((0, 5), dtype=np.float32)
        det_queue.put((frame, detections))
    cap.release()


#Thread 2 : le tracker
#récupère chaque item de la file pour la traiter.
def tracker_worker(det_queue, output_path, fps, width, height):
    tracker = create_grape_tracker(camera_speed_kmh=3.5,fps=30, min_hits=2)
    if video_output: 
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    unique_ids = set()
    while True:
        item = det_queue.get()
        if item is None:
            break
        frame, detections = item
        trackers = tracker.update(detections)
        for d in trackers:
            x1, y1, x2, y2, track_id = d
            unique_ids.add(int(track_id))
            if video_output:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'ID {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if video_output:
            out.write(frame)

    if video_output:
        out.release()
        print(f"Saved {output_path}")

    print(f"Final unique count: {len(unique_ids)}")


#Threads launcher
def process_video_parallel(weights, video_path, output_dir, img_size, device_str):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    
    out_path = Path(output_dir) / f"det_{Path(video_path).name}"

    stride = 32  # typical for yolov5 small
    names = ["grape"]  

    det_queue = mp.Queue(maxsize=20)  # tune maxsize depending on RAM

    p_det = mp.Process(target=detector_worker, args=(video_path, img_size, stride, names, weights, device_str, det_queue))
    p_track = mp.Process(target=tracker_worker, args=(det_queue, out_path, fps, width, height))

    p_det.start()
    p_track.start()

    p_det.join()
    p_track.join()


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Video path")
    parser.add_argument("--weights", default="yolov5s.pt")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--video_output",type=str2bool, default=False)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    process_video_parallel(args.weights, args.input_path, args.output_dir, args.img_size, '')
    #global video_output 
    video_output = args.video_output

if __name__ == "__main__":
    mp.set_start_method('spawn')  # safer for Jetson boards
    main()
