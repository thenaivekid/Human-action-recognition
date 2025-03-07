import uvicorn
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import subprocess
from pyngrok import ngrok
import nest_asyncio
from pydantic import BaseModel
from utils import predict_video_label, download_video
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

import pytorchvideo.data#import torchvision.transforms.functional as F_t should should replace
#----> 9 import torchvision.transforms.functional_tensor as F_t

from torchvision.transforms import Compose


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)


from safetensors.torch import load_file

### for ucf101
# model_name = "thenaivekid/videomae-base-finetuned-ucf101-subset"
# model = VideoMAEForVideoClassification.from_pretrained(model_name)
# image_processor = VideoMAEImageProcessor.from_pretrained(model_name)


### for finetuned ntu
class_labels = [
    "drink water", "brush teeth", "pick up", "reading", "writing", 
    "cheer up", "jump up", "phone call", "taking a selfie", "salute"
]
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,#true when finetuning already finetuned ckt
)
ckpt = "/home/thenaivekid/har/backend/checkpoint/model.safetensors"
state_dict = load_file(ckpt)

model.load_state_dict(state_dict)

# mean = image_processor.image_mean
# std = image_processor.image_std
# if "shortest_edge" in image_processor.size:
#     height = width = image_processor.size["shortest_edge"]
# else:
#     height = image_processor.size["height"]
#     width = image_processor.size["width"]
# resize_to = (height, width)

# num_frames_to_sample = model.config.num_frames
# sample_rate = 4
# fps = 30
# clip_duration = num_frames_to_sample * sample_rate / fps


# val_transform = Compose(
#                 [
#                     Lambda(lambda x: x / 255.0),
#                     Normalize(mean, std),
#                     Resize(resize_to),
#                 ]
#             )
      

app = FastAPI()
# Allow all CORS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# # Simulated function: Replace with your actual prediction function
# def predict_video_label(video_path: str):
#     """ Dummy function to simulate video classification """
#     return "Dummy Label"

# Directory to save uploaded videos
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    print(file_path, "ashok file path")
    # Save the uploaded video file
    with file_path.open("wb") as buffer:
        buffer.write(await file.read())

    # Predict label
    label = predict_video_label(str(file_path), model, image_processor)
    return {"filename": file.filename, "label": label}

class InputLink(BaseModel):
    link: str
@app.post("/upload_link/")
async def upload_video_link(input_link: InputLink):
    file_path = download_video(input_link.link)
    print(file_path, "ashok file path")

    label = predict_video_label(str(file_path), model, image_processor)

    return {"filename": file_path, "label": label}


if __name__=="__main__":
    import os
    os.environ["NGROK_AUTHTOKEN"] = "2rCbW45ffaTmVf03HbMluTUNCv1_4uD242hh56wg9SvHorrNR"
    import uvicorn
    ngrok_tunnel=ngrok.connect(8000)
    print('Public_URL: ',ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app)

