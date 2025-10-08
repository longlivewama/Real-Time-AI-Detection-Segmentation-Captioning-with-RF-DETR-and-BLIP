!pip install -U rfdetr supervision pillow opencv-python tqdm -q

from google.colab import files
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

uploaded = files.upload()
video_path = list(uploaded.keys())[0]

model = RFDETRBase(model_id="rfdetr-seg-preview")

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_path = "output_segmented.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

mask_annotator = sv.MaskAnnotator(
    color=sv.ColorPalette.ROBOFLOW,
    opacity=0.6
)
label_annotator = sv.LabelAnnotator(text_color=sv.Color.WHITE)

for _ in tqdm(range(frame_count), desc="Processing video..."):
    ret, frame = cap.read()
    if not ret:
        break

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = model.predict(image, threshold=0.5)

    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections)

    frame_out = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    out.write(frame_out)

cap.release()
out.release()
print("✅ Segmentation video saved:", output_path)

from IPython.display import Video
Video(output_path, embed=True)
