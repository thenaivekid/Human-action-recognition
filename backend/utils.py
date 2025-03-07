import requests
from datetime import datetime
def download_video(video_url, output_path="uploads/"):
    filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    output_path = output_path + filename
    response = requests.get(video_url, stream=True)

    if response.status_code == 200:
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Video downloaded successfully and saved to {output_path}")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")
    return output_path

import av
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from huggingface_hub import hf_hub_download

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def predict_video_label(video_path, model, image_processor):
    """
    Predict the label for a given video using VideoMAE model.
    
    Args:
        video_path (str): Path to the video file
        model_name (str, optional): Hugging Face model name. Defaults to a Kinetics-finetuned model.
    
    Returns:
        str: Predicted video action/class label
    """
    # Set random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Open the video container
    container = av.open(video_path)
    
    # Get total number of frames in the video
    total_frames = container.streams.video[0].frames
    
    # Sample 16 frames
    clip_len = 16
    indices = sample_frame_indices(
        clip_len=clip_len, 
        frame_sample_rate=1, 
        seg_len=total_frames
    )
    
    # Read the sampled frames
    video = read_video_pyav(container, indices)
    # Prepare video for the model
    inputs = image_processor(list(video), return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class
    predicted_class_idx = outputs.logits.argmax(dim=1).item()
    
    # Get the class label
    predicted_label = model.config.id2label[predicted_class_idx]
    
    return predicted_label

# Example usage
def main():
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    from safetensors.torch import load_file

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
    video_path = "/home/thenaivekid/har/backend/uploads/S002C001P007R001A003_rgb.avi"
    
    # Predict the label
    label = predict_video_label(video_path, model, image_processor)
    print(f"Predicted video action: {label}")

if __name__ == "__main__":
    main()