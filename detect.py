import cv2
import numpy as np

# Load the image
img = cv2.imread('image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascades classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Calculate the mean brightness of the image
mean_brightness = np.mean(gray)

# Determine whether the image was taken in the morning or at night
if mean_brightness > 100:
    time_of_day = 'morning'
else:
    time_of_day = 'night'

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the time of day and the number of faces detected
print(f'This image was taken in the {time_of_day}.')
print(f'{len(faces)} faces were detected in the image.')
