#import the ai face detector library
import cv2

from random import randrange

# Load pre-trained data on face frontals from opencv, 
# Classifier is a detector eg: detect faces
# Casscade is an alogorithim i.e. algorthim to identify what a face is
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam, 0 automatically goes to default webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:

    # this reads the current fram
    # check if reading the webcam was a success or not, then gets the frame
    successful_frame_read, frame = webcam.read()

    # Must convert to black and white/grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    # MultiScale is to detect a face on multiple scales eg: peoples eyes can be different sizes so we want to be detect all types of eyes
    # location of faces in image stored as an array of coordinates eg: [ face 1, face 2, face 3 ]
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw a rectangle around the face
    # for each face found in the image
    # so first get the image, then takes upper left hand coordinate, then the upper right hand coordinate, 
    # then the color randrage used to get different color for each face border, then how thick the rectangle is
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

    cv2.imshow('Face Detector', frame)

    # Waits 1 frame then randomly a key is pressed to continue.
    #cv2.waitKey(1)

    # If Q is pressed exit the app
    key = cv2.waitKey(1)
    if key==81 or key ==113:
        break

#Release the webcam video
webcam.release()
"""
    print(face_coordinates)

    # Draw a rectangle around the face
    # for each face found in the image
    # so first get the image, then takes upper left hand coordinate, then the upper right hand coordinate, 
    # then the color randrage used to get different color for each face border, then how thick the rectangle is
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256),randrange(256),randrange(256)), 2)

    #imshow function = image show
    cv2.imshow('Face Detector', frame)

    # Wait until a key is pressed.
    cv2.waitKey()

"""