import cv2
import face_recognition

def detect_and_display_faces(image_path):
    """
    Loads an image, detects face locations using HOG/CNN models,
    draws bounding boxes, and displays the result.
    """
    # Load image from path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        return

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face bounding boxes: (top, right, bottom, left)
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    print(f"Detected {len(face_locations)} face(s) in the image.")

    # Annotate image
    for i, (top, right, bottom, left) in enumerate(face_locations, start=1):
        # Draw bounding box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw label index
        cv2.putText(
            image, f"Face #{i}", (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # Display window
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test face detection on a sample image
    detect_and_display_faces("sample.jpg")
