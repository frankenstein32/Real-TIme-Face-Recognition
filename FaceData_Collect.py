# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import numpy as np
import cv2

# Instantiate the Object of VideoCapture
cap = cv2.VideoCapture(0)

# Instantiation of a Haarcascade Classifier
face_cascade = cv2.CascadeClassifier("haarcascade.xml")

# List to store frame's Coordinates
face_data = []
skip = 0
path_name = "./data/"
# face_section = Nones

# Input Name of the User
file_name = input("Enter the Name of the User:")

while True:

	# Reading Frame by Frame
	ret, frame = cap.read()

	#check if the Frame is detected or not
	if ret == False:
		continue

	#Convering Color Frame to Gray Frame
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

	# Detect the Face and return the coordinates in list form
	face_points = face_cascade.detectMultiScale(gray_frame,1.3,5)
	if len(face_points)==0:
		continue

	# Sort the list in reverse Order so that we find the biggest frame
	face_points = sorted(face_points,key = lambda f:f[2]*f[3])

	# Largest Frame is at first Position
	for face in face_points[-1:]:

		x,y,w,h = face

		# Drawing Rectangle
		cv2.rectangle(frame,(x,y),(x+w,y+h),(25,0,0),2)

		# Extract the Crop out Face
		offset = 10

		# Creating A Frame of padding equal to off-set
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]

		# Resize the Section into (100 X 100)
		face_section = cv2.resize(face_section,(100,100))

		# Incrementing the Skip
		skip += 1

		# Capturing every 10th Frame
		if skip % 10 == 0:

			# Adding the Section into List
			face_data.append(face_section)

			# Printing Length
			print(len(face_data))
 

	# Display the Frame
	cv2.imshow('frame',frame)

	#Displaying the face Section
	cv2.imshow('face Section',face_section)

	# Exit when q is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Creating Numpy Array
face_data = np.asarray(face_data)

# Flatened
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Saving data
np.save(path_name+file_name+'.npy',face_data)
print("Data Saved Successfully")

# Destroy and Release the Objects
cap.release()
cv2.destroyAllWindows()




