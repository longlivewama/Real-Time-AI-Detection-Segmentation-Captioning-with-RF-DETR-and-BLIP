!pip install -U transformers torch torchvision pillow tqdm opencv-python -q

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import cv2
from PIL import Image
from google.colab import files

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

video_path = "/content/output_detected (2).mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "/content/output_captioned_blip.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
thickness = 2
padding = 10
alpha = 0.6

captions = []
frame_count = 0
sample_rate = fps  

print("🔍 Generating captions with BLIP-2...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % sample_rate == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        captions.append(caption)
        current_caption = caption
    else:
        current_caption = captions[-1] if captions else ""

    text_size, _ = cv2.getTextSize(current_caption, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = 30, height - 40
    rect_x1, rect_y1 = x - padding, y - text_h - padding
    rect_x2, rect_y2 = x + text_w + padding, y + padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, current_caption, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

print("\n✅ Captioned video saved to:", output_path)
print("📝 Video Description Summary:")
print(" | ".join(captions))

files.download(output_path)
