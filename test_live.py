import cv2
import time

def main():
    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    # Initialize counter and last update time
    counter = 0
    last_update_time = time.time()
    
    while True:
        # Read frame from video
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Get current time
        current_time = time.time()
        
        # Update counter every second
        if current_time - last_update_time >= 1.0:
            counter += 1
            last_update_time = current_time
        
        # Draw rectangle and counter text
        cv2.rectangle(frame, (10, 10), (200, 70), (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {counter}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Live Video Counter', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
