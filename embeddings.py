import numpy as np
import face_recognition

def extract_face_encodings(image_path):
    """
    Extracts 128-dimensional feature embedding vectors for all faces in an image.
    """
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings

def calculate_similarity(encoding1, encoding2):
    """
    Calculates Euclidean distance and similarity percentage between two face encodings.
    Lower Euclidean distance means greater similarity.
    """
    distance = np.linalg.norm(encoding1 - encoding2)
    # Convert Euclidean distance to approximate percentage similarity
    similarity_score = max(0.0, (1.0 - distance)) * 100
    return distance, similarity_score

def compare_two_faces(img_path_a, img_path_b, threshold=0.6):
    """
    Compares two face images to check if they belong to the same person.
    """
    encodings_a = extract_face_encodings(img_path_a)
    encodings_b = extract_face_encodings(img_path_b)

    if not encodings_a or not encodings_b:
        print("Error: Could not extract face encodings from one or both images.")
        return False

    distance, similarity = calculate_similarity(encodings_a[0], encodings_b[0])
    is_match = distance <= threshold

    print(f"Euclidean Distance: {distance:.4f}")
    print(f"Similarity Match: {similarity:.2f}%")
    print(f"Same Person? {'YES' if is_match else 'NO'}")

    return is_match

if __name__ == "__main__":
    # Test comparison between two images
    compare_two_faces("person1.jpg", "person2.jpg")
