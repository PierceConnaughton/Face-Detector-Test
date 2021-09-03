#import the ai face detector image
import cv2

# Load pre-trained data on face frontals from opencv, 
# Classifier is a detector eg: detect faces
# Casscade is an alogorithim i.e. algorthim to identify what a face is
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect a face in
# imread function = image read

#img = cv2.imread('FaceOne.png')
img = cv2.imread('MultiFace.png')

# Must convert to black and white/grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
# MultiScale is to detect a face on multiple scales eg: peoples eyes can be different sizes so we want to be detect all types of eyes
# location of faces in image stored as an array of coordinates eg: [ face 1, face 2, face 3 ]
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coordinates)

# Draw a rectangle around the face
# for each face found in the image
# so first get the image, then takes upper left hand coordinate, then the upper right hand coordinate, then the color, then how thick the rectangle is
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,0), 2)

#imshow function = image show
cv2.imshow('Face Detector', img)

# Wait until a key is pressed.
cv2.waitKey()

print("Code Completed")