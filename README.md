

# VisionSentinel: Real-Time AI Detection, Segmentation & Captioning with RF-DETR and BLIP

---

## 📌 Project Description

**VisionSentinel: Real-Time AI Detection, Segmentation & Captioning with RF-DETR and BLIP** is a high-performance AI framework designed to analyze videos and images in **real-time**, combining **state-of-the-art object detection, instance segmentation, and contextual captioning**.

This project performs advanced basketball video analysis using AI techniques:

* **Player Segmentation** – Detects and segments basketball players from the video.
* **Ball Detection** – Identifies the basketball in the video.
* **Team Identification** – Differentiates between two teams.
* **Court Segmentation** – Highlights different areas of the court.
* **Video Captioning** – Generates textual descriptions of key scenes automatically using BLIP (Bootstrapped Language-Image Pretraining).

**Technologies Used:**

* **RF-DETR** – Real-time object segmentation for players, ball, and court areas.
* **Supervision Library** – For mask and label annotation.
* **BLIP (Salesforce)** – Automatic video/image captioning.
* **OpenCV & PIL** – Video and image processing.
* **Python & PyTorch** – Core programming and deep learning framework.

**Key Features:**

* Real-time multi-object detection for players and ball.
* Pixel-perfect segmentation of teams and court areas.
* Contextual captions describing video actions and events.
* Fully annotated video output with segmentation masks, bounding boxes, and textual overlays.
* Video summarization with readable textual descriptions.
* Optimized for GPU acceleration and batch processing.

---

### 📷 RF-DETR Overview Image

![RF-DETR and BLIP](RF-DETR%20and%20BLIP.png)

---

## 🛠 Installation

Requires **Python >= 3.9**.

```bash
pip install -U rfdetr supervision pillow opencv-python tqdm transformers torch torchvision
```

For latest cutting-edge features, install RF-DETR from source:

```bash
pip install git+https://github.com/roboflow/rf-detr.git
```

---

## 🚀 Usage

### 1️⃣ Object Detection & Segmentation

* Uses `rfdetr-seg-preview` for high-speed, pixel-accurate instance segmentation.
* Annotates frames/images with advanced masks and labels.

```python
from rfdetr import RFDETRBase
import supervision as sv
from PIL import Image

model = RFDETRBase(model_id="rfdetr-seg-preview")
model.optimize_for_inference()

image = Image.open("frame.jpg")
detections = model.predict(image, threshold=0.5)

annotated_image = sv.MaskAnnotator().annotate(image.copy(), detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)
```

---

### 2️⃣ Contextual Video/Image Captioning with BLIP

* Generates descriptive captions and overlays them on video frames.

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open("frame.jpg")
inputs = processor(images=image, return_tensors="pt").to(device)
out_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(out_ids[0], skip_special_tokens=True)
print("Caption:", caption)
```

---

## ⚡ Advanced Features

* Real-time multi-object detection with **RF-DETR Seg (Preview)**.
* Pixel-perfect instance segmentation optimized for dynamic scenes.
* Contextual caption generation using BLIP for per-frame insight.
* Fully annotated outputs with bounding boxes, masks, and textual context.
* High-speed inference optimized for GPU and batch processing.

---

## 📂 Project Structure

```
vision-sentinel/
│
├─ videos/                  # Input videos/images
├─ output/                  # Segmented & captioned outputs
├─ main_segmentation.py     # RF-DETR segmentation script
├─ main_captioning.py       # BLIP captioning script
├─ RF-DETR and BLIP.png     # Overview image
├─ README.md
```

---

## 📈 References & Acknowledgements

* [RF-DETR](https://github.com/roboflow/rf-detr) – Real-time SOTA object detection and segmentation.
* [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) – Contextualized image/video captioning.
* [Supervision](https://github.com/roboflow/supervision) – Advanced annotation utilities.

**Citation:**

```bibtex
@software{rf-detr,
  author = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei and Ramanan, Deva and Peri, Neehar},
  license = {Apache-2.0},
  title = {RF-DETR},
  howpublished = {\url{https://github.com/roboflow/rf-detr}},
  year = {2025},
  note = {SOTA Real-Time Object Detection Model}
}
```
