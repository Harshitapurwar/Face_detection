import cv2
import os
import glob

def generate_dataset(user_id):
    # Load the Haar Cascade for face detection
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    # Function to crop face from an image
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        # Check if faces are detected
        if len(faces) == 0:
            return None
        
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face

    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Check if data for this user_id already exists
    existing_files = glob.glob(f"data/user.{user_id}.*.jpg")
    if existing_files:
        permission = input(f"Data for user ID {user_id} already exists. Do you want to overwrite it? (y/n): ").strip().lower()
        if permission != 'y':
            print("Operation canceled.")
            return

    # Capture video feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    img_id = 0  # Start image numbering from 0 for each user

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break

        cropped_face = face_cropped(frame)
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{user_id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Cropped face", face)
        
        # Break loop when Enter is pressed or 200 images are captured
        if cv2.waitKey(1) == 13 or img_id == 200:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collecting samples for user {user_id} is completed.")

# Prompt user for ID input
user_id = input("Enter user ID: ")
generate_dataset(user_id)


# face_classifier
import os
import cv2
from PIL import Image #pip install pillow
import numpy as np    # pip install numpy
 
def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
     
    faces = []
    ids = []
     
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
         
        faces.append(imageNp)
        ids.append(id)
         
    ids = np.array(ids)
     
    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
train_classifier("data")

# trying to match faces
import os
import cv2
from PIL import Image
import numpy as np

# Define the boundary drawing function
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, pred = clf.predict(gray_img[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 300))

        if confidence > 80:
            if id == 1:
                cv2.putText(img, "Lincy", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id == 2:
                cv2.putText(img, "Tanya", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    
    return img

# Load face cascade and recognizer
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Start video capture
video_capture = cv2.VideoCapture(0)  # Change to 0 if 1 doesn’t work

if not video_capture.isOpened():
    print("Error: Could not open video source.")
else:
    while True:
        ret, img = video_capture.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        if img is not None:
            img = draw_boundary(img, faceCascade, 1.3, 6, (255, 255, 255), "Face", clf)
            cv2.imshow("Face Detection", img)
        else:
            print("Warning: Empty frame received.")
        
        # Break on pressing Enter (ASCII 13)
        if cv2.waitKey(1) == 13:
            break

    # Release the capture and close any OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()
