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