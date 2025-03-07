import cv2
import time
import numpy as np
import torch
from collections import deque
from safetensors.torch import load_file
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# Import utility functions from utils.py
from backend.utils import sample_frame_indices

def main():
    # Initialize class labels and model
    class_labels = [
        "drink water", "brush teeth", "pick up", "reading", "writing", 
        "cheer up", "jump up", "phone call", "taking a selfie", "salute"
    ]
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Load model and image processor
    model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    
    # Load the fine-tuned checkpoint
    ckpt = "/home/thenaivekid/har/backend/checkpoint/model.safetensors"
    try:
        state_dict = load_file(ckpt)
        model.load_state_dict(state_dict)
        print("Model checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    # Initialize variables
    counter = 0
    last_update_time = time.time()
    last_inference_time = time.time()
    inference_interval = 2.0  # Run inference every 2 seconds
    
    # Set up frame buffer for inference
    num_frames_to_sample = 16
    frame_buffer = deque(maxlen=30)  # Store 30 frames (1 second at 30fps)
    current_action = "Initializing..."
    
    while True:
        # Read frame from video
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Add frame to buffer - convert to RGB since models expect RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb_frame)
        
        # Get current time
        current_time = time.time()
        
        # Update counter every second
        if current_time - last_update_time >= 1.0:
            counter += 1
            last_update_time = current_time
        
        # Run inference periodically if we have enough frames
        if current_time - last_inference_time >= inference_interval and len(frame_buffer) >= num_frames_to_sample:
            last_inference_time = current_time
            
            try:
                # Sample frames from the buffer
                frames_array = list(frame_buffer)
                total_frames = len(frames_array)
                
                # Get indices of frames to sample using util function
                indices = sample_frame_indices(
                    clip_len=num_frames_to_sample,
                    frame_sample_rate=1,
                    seg_len=total_frames
                )
                
                # Extract the sampled frames and stack them into a numpy array
                # This mimics the output format of read_video_pyav
                video = np.stack([frames_array[i] for i in indices])
                
                # Process frames with the image processor (exactly as in utils.py)
                inputs = image_processor(list(video), return_tensors="pt")
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                
                # Perform inference (exactly as in utils.py)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get predicted class
                predicted_class_idx = outputs.logits.argmax(dim=1).item()
                current_action = id2label[predicted_class_idx]
                print(f"Predicted action: {current_action}")
                
            except Exception as e:
                print(f"Inference error: {e}")
        
        # # Draw rectangle and counter text
        # cv2.rectangle(frame, (10, 10), (200, 70), (0, 255, 0), 2)
        # cv2.putText(frame, f"Count: {counter}", (20, 50), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the predicted action
        cv2.rectangle(frame, (10, 80), (400, 140), (0, 0, 255), 2)
        cv2.putText(frame, f"Action: {current_action}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Live Video Action Recognition', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
