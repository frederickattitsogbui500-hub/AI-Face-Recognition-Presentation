import os
import cv2
import numpy as np
import face_recognition

class FaceRecognitionAgent:
    """
    AI Agent for detecting, encoding, registering, and recognizing faces.
    """
    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load pre-registered faces upon initialization
        self.load_known_faces()

    def load_known_faces(self):
        """Loads and encodes all face images stored in the known_faces directory."""
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"[Agent] Created directory '{self.known_faces_dir}'. Add images here to register users.")
            return

        print("[Agent] Loading known faces...")
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    # Register the first detected face in the image
                    name = os.path.splitext(filename)[0].replace("_", " ").title()
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(name)
                    print(f" -> Registered: {name}")
                else:
                    print(f" -> Warning: No face found in {filename}")

    def register_new_face(self, image_path, person_name):
        """Registers a single new face into memory dynamically."""
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(person_name)
            print(f"[Agent] Successfully registered new user: {person_name}")
            return True
        else:
            print(f"[Agent] Error: No face detected in {image_path}")
            return False

    def process_frame(self, frame, tolerance=0.5):
        """
        Performs face detection, extraction, embedding matching, 
        and bounding box drawing on a video frame.
        """
        # Convert frame BGR (OpenCV format) to RGB (face_recognition format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # 2. Extract face embeddings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"

            if len(self.known_face_encodings) > 0:
                # 3. Compute Euclidean distance against stored face vectors
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=tolerance
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # 4. Annotate image with bounding boxes and names
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw bounding box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw background label box
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            
            # Overlay name label
            cv2.putText(
                frame, name, (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        return frame

    def run_live_recognition(self):
        """Starts live webcam feed for real-time face recognition."""
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("[Agent] Error: Could not open camera.")
            return

        print("[Agent] Live Face Recognition active. Press 'q' to exit.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Process the frame through the agent pipeline
            processed_frame = self.process_frame(frame)

            # Display real-time output window
            cv2.imshow("AI Face Recognition Agent", processed_frame)

            # Quit window on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize the Agent
    agent = FaceRecognitionAgent(known_faces_dir="known_faces")
    
    # Run Real-Time Recognition
    agent.run_live_recognition()
