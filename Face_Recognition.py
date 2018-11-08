# Recognise Faces using some classification algorithm - like Logistic, KNN, SVM etc.


# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

import numpy as np 
import cv2
import os


##########################################################################

# Distance Function
def dist(x1,x2):

	return np.sqrt(((x1-x2)**2).sum())

def KNN(train,test,k = 5):

	# List that will store the tuples of (distance , prediction)
	distances = []
	
	# Loop on Each Element
	for ix in range(train.shape[0]):

		# ix data point 
		x_data = train[ix,:-1]

		# ix label
		y_data = train[ix,-1]

		# Euclidian Distance between testing point and the training point
		d = dist(x_data,test)

		# Creating list of tuples
		distances.append((d,y_data))

	# Sorting array so that closer points come in starting
	dk = sorted(distances,key = lambda f : f[0])[:k] #Only taking first K values

	# Creating Unique array to find max count of every value in the sorted list
	temp = np.unique(dk,return_counts= True)

	# finding the index of value having max count
	index = np.argmax(temp[1])

	# return that pred
	return temp[0][index]

##########################################################################


# path where all of the data is Saved
dataset_path = './data/'

# Storing the X - values
face_data = []

# Storing the Y-values
labels = []

# Class Id that will be assing to each class
class_id = 0

# Dict containing class-id versus names
names = {}

# Looping on each file
for fx in os.listdir(dataset_path):

	# Only taking files ending with npy
	if fx.endswith('.npy'):

		# storing name of the File in a dict of key class_id
		names[class_id] = fx[:-4]

		# Printing File name
		print("Loaded "+fx)

		# Loading the values that we have stored
		data_item = np.load(dataset_path+fx)

		#appending the values into the face data list
		face_data.append(data_item)

		# creating label for all values of same class_id
		target = class_id*np.ones((data_item.shape[0],))

		# Incrementing class_id for next iteration
		class_id += 1

		#appending the label to labels list 
		labels.append(target)

# creating the face_data eligible for making data set
face_dataset = np.concatenate(face_data,axis = 0)
face_labels = np.concatenate(labels,axis = 0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis = 1)
print(trainset.shape)


cap = cv2.VideoCapture(0)

skip = 0

face_cascade = cv2.CascadeClassifier("haarcascade.xml")

while True:

	ret, frame = cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

	if ret is False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	if len(faces) == 0:
		continue

	for face in faces:
		x,y,w,h = face 
		

		offset = 10

		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		# pred for the given testing value
		out = KNN(trainset,face_section.flatten())

		# Finding the name cprresponding to the pred
		pred = names[int(out)]

		# Writing the name on the video
		cv2.putText(frame,pred,(x , y - 10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

		# Creating the Frame
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		# Display Frame
	cv2.imshow('frame',frame)

	# Exit if q is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Releasing and Destroying the objects
cap.release()
cv2.destroyAllWindows()