import cv2
import numpy as np
from sklearn.svm import SVC

# Dummy training data
X = np.random.rand(20, 4096)
y = np.array(['Palm', 'Fist'] * 10)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X, y)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Define region of interest
    roi = frame[100:300, 100:300]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Resize image
    resized = cv2.resize(gray, (64, 64))

    # Flatten image
    flattened = resized.flatten().reshape(1, -1)

    # Predict gesture
    prediction = model.predict(flattened)

    # Draw rectangle
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    # Display prediction
    cv2.putText(
        frame,
        f'Gesture: {prediction[0]}',
        (100, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()