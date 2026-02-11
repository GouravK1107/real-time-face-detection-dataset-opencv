import cv2
import os

# Loading Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

name = input("Enter name: ")

# Create dataset folder
dataset_path = f"dataset/{name}"
os.makedirs(dataset_path, exist_ok=True)

# Continue numbering, if folder already has images
img_count = len(os.listdir(dataset_path))
max_images = 10

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# To show login only once
login_triggered = False  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # Save image
        if img_count < max_images:
            img_count += 1
            file_name = f"{dataset_path}/face_{img_count}.jpg"
            cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        login_triggered = True

    # Display text
    if login_triggered:
        cv2.putText(frame, f"Login Successful - {name}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    cv2.putText(frame, f"Images Saved: {img_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    cv2.imshow("Face Capture", frame)


    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

    if img_count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
