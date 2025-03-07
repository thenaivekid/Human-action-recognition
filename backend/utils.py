# import torch
# from pytorchvideo.data.encoded_video import EncodedVideo
# import torch
# import requests
# from datetime import datetime
# import cv2
# from PIL import Image
# import numpy as np
# def load_and_sample_frames(video_path, clip_len = 16,):
#     """
#     Load video and sample clip_len frames uniformly from the entire video
#     """
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Handle videos with fewer frames than clip_len
#     if total_frames <= clip_len:
#         frame_indices = list(range(total_frames))
#         # Repeat the last frame if necessary
#         while len(frame_indices) < clip_len:
#             frame_indices.append(total_frames - 1)
#     else:
#         # Sample clip_len frames uniformly
#         frame_indices = np.linspace(0, total_frames - 1, clip_len, dtype=int)
    
#     frames = []
#     for idx in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         print(f"ret {ret}")
#         if ret:
#             # Convert BGR to RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = Image.fromarray(frame)
#             frames.append(frame)
#         else:
#             # If reading fails, create a blank frame
#             frames.append(Image.new('RGB', (1024, 1024), color=0))
    
#     cap.release()
#     return frames


# def predict_video_label(file_path, model, val_transform, id2label):
#     """
#     Predicts the label of a video using the provided model.

#     Args:
#         file_path (str): Path to the video file.
#         model (torch.nn.Module): Pre-trained model for video classification.
#         val_transform (torchvision.transforms.Compose): Transformations to apply to the video.
#         id2label (dict): Dictionary mapping class IDs to labels.

#     Returns:
#         str: Predicted label for the video.
#     """
#     # Check for CUDA availability
#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "cuda"
#     model = model.to(device)
#     model.eval()
#     print(device, "ashok using device")

#     # Load the video
#     print(file_path, "ashok predict label")
#     video_frames = load_and_sample_frames(file_path)
#     # clip_duration = video.duration

#     # # Sample a clip from the video
#     # clip = video.get_clip(start_sec=0, end_sec=clip_duration)

#     # Apply the transformations
#     frames_tensor = torch.stack([val_transform(frame) for frame in video_frames])
        
#         # Shape is now [16, 3, 1024, 1024] - we need [3, 16, 1024, 1024]
#     frames_tensor = frames_tensor.permute(1, 0, 2, 3).to(device)  # [T, C, H, W] -> [C, T, H, W]
#     # transformed_clip = transformed_clip["video"].unsqueeze(0).to(device)  # Add batch dimension and move to device

#     # Perform inference
#     with torch.no_grad():
#         output = model(frames_tensor)  # Permute to (B, C, T, H, W)
#         pred_id = torch.argmax(output.logits, dim=1).item()

#     return id2label[pred_id]



# def download_video(video_url, output_path="uploads/"):
#     filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
#     output_path = output_path + filename
#     response = requests.get(video_url, stream=True)

#     if response.status_code == 200:
#         with open(output_path, "wb") as file:
#             for chunk in response.iter_content(chunk_size=1024):
#                 file.write(chunk)
#         print(f"Video downloaded successfully and saved to {output_path}")
#     else:
#         print(f"Failed to download video. Status code: {response.status_code}")
#     return output_path
# if __name__ == "__main__":
    
#     pass
#     from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

#     import pytorchvideo.data#import torchvision.transforms.functional as F_t should should replace
#     #----> 9 import torchvision.transforms.functional_tensor as F_t

#     from torchvision.transforms import Compose


#     from pytorchvideo.transforms import (
#         ApplyTransformToKey,
#         Normalize,
#         RandomShortSideScale,
#         RemoveKey,
#         ShortSideScale,
#         UniformTemporalSubsample,
#     )

#     from torchvision.transforms import (
#         Compose,
#         Lambda,
#         RandomCrop,
#         RandomHorizontalFlip,
#         Resize,
#     )




#     model_name = "thenaivekid/videomae-base-finetuned-ucf101-subset"
#     model = VideoMAEForVideoClassification.from_pretrained(model_name)
#     image_processor = VideoMAEImageProcessor.from_pretrained(model_name)


#     mean = image_processor.image_mean
#     std = image_processor.image_std
#     if "shortest_edge" in image_processor.size:
#         height = width = image_processor.size["shortest_edge"]
#     else:
#         height = image_processor.size["height"]
#         width = image_processor.size["width"]
#     resize_to = (height, width)

#     num_frames_to_sample = model.config.num_frames
#     sample_rate = 4
#     fps = 30
#     clip_duration = num_frames_to_sample * sample_rate / fps



#     val_transform = Compose(
#                 [
#                     Lambda(lambda x: x / 255.0),
#                     Normalize(mean, std),
#                     Resize(resize_to),
#                 ]
#             )
      


#     video_path = "/home/thenaivekid/har/backend/uploads/v_BenchPress_g05_c02.avi"
#     print("predicted action: ",predict_video_label(video_path, model, val_transform, model.config.id2label))
#     # download_video(video_url="https://utfs.io/f/atDsyOESFTxlddd1bzke1mJ6pIFbfKrOEMkgH7vz3hPxianZ")


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
    # You can replace this with your own video path
    # video_path = hf_hub_download(
    #     repo_id="nielsr/video-demo", 
    #     filename="eating_spaghetti.mp4", 
    #     repo_type="dataset"
    # )
    video_path = "/home/thenaivekid/har/backend/uploads/v_BenchPress_g05_c02.avi"
    
    # Predict the label
    label = predict_video_label(video_path)
    print(f"Predicted video action: {label}")

if __name__ == "__main__":
    main()