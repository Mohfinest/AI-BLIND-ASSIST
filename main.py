import torch
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
import tempfile
from collections import defaultdict
import gradio as gr
import time

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Target classes for detection
TARGET_CLASSES = [
        # Animals
    'person', 'cat', 'dog', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe',
    'lion', 'tiger', 'rabbit', 'monkey', 'panda', 'kangaroo', 'deer', 'squirrel', 'fox', 'penguin',

    # Vehicles
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'airplane', 'helicopter', 'train', 'van',
    'ambulance', 'fire truck', 'police car', 'scooter', 'skateboard', 'tractor',

    # Household Items
    'chair', 'table', 'sofa', 'cap', 'bucket' 'bed', 'harmmer', 'kettle', 'iron', 'pillow', 'blanket', 'lamp', 'fan', 'microwave', 'oven',
    'refrigerator', 'stove', 'sink', 'faucet', 'mirror', 'toilet', 'bathtub', 'shower', 'curtain', 'door',

    # Electronics
    'TV', 'remote', 'cell phone', 'tablet', 'laptop', 'monitor', 'keyboard', 'mouse', 'printer', 'camera',
    'headphones', 'speaker', 'microphone', 'router', 'smartwatch', 'game console', 'VR headset', 'projector',

    # Tools
    'hammer', 'screwdriver', 'wrench', 'pliers', 'saw', 'drill', 'ladder', 'toolbox', 'shovel', 'rake',

    # Kitchenware
    'plate', 'bowl', 'spoon', 'fork', 'knife', 'cup', 'mug', 'glass', 'saucepan', 'frying pan',
    'pot', 'spatula', 'whisk', 'ladle', 'tongs', 'colander', 'cutting board', 'mixing bowl',

    # Food Items
    'apple', 'banana', 'orange', 'grape', 'watermelon', 'strawberry', 'cherry', 'pineapple', 'peach', 'kiwi',
    'lemon', 'lime', 'carrot', 'potato', 'tomato', 'lettuce', 'cucumber', 'pepper', 'onion', 'garlic',

    # Clothing
    't-shirt', 'shirt', 'pants', 'jeans', 'shorts', 'jacket', 'coat', 'sweater', 'hoodie', 'dress',
    'skirt', 'hat', 'cap', 'belt', 'socks', 'shoes', 'boots', 'sandals', 'sneakers', 'scarf',

    # Office Supplies
    'pen', 'pencil', 'notebook', 'paper', 'stapler', 'tape', 'scissors', 'marker', 'ruler', 'calendar',

    # Miscellaneous
    'book', 'magazine', 'map', 'toy', 'teddy bear', 'doll', 'robot', 'trophy', 'coin', 'passport'

]

# Timer to control TTS frequency
last_audio_time = 0
AUDIO_COOLDOWN = 30  # Cooldown time in seconds between audio feedback

# Function to detect objects in the image and generate a description
def detect_objects(image):
    global last_audio_time
    img = np.array(image)

    # Run object detection
    results = model(img)
    detections = results.pred[0].cpu().numpy()  # Get detections

    # Initialize object count
    object_count = defaultdict(int)
    for *box, conf, cls in detections:
        label = model.names[int(cls)]
        if label in TARGET_CLASSES and conf > 0.5:  # Confidence threshold
            object_count[label] += 1

    # Generate description text
    description = ', '.join([f"{count} {obj}(s)" for obj, count in object_count.items()])
    if not description:
        description = "No objects detected."

    # Generate TTS audio if objects were detected, respecting cooldown
    audio_path = None
    if description != "No objects detected.":  # Only create audio for meaningful detections
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts = gTTS(text=description, lang='en')
            tts.save(fp.name)
            audio_path = fp.name

    return results.render()[0], audio_path  # Rendered image and audio path

# Create Gradio interface
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(streaming=True, label="Webcam Feed"),
    outputs=[gr.Image(type="pil"), gr.Audio(type="filepath")],
    live=True,
    description="Blind Assistant"
)

# Launch Gradio app
iface.launch()
