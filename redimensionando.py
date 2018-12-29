from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.image as img
from skimage.transform import resize
import matplotlib.pyplot as plt
import scipy.misc
from os import listdir
from skimage import filters
from os.path import isfile, join
from PIL import Image, ImageFilter
from scipy import ndimage
import numpy as np
import cv2
from sklearn import datasets, svm, metrics
#np.set_printoptions(threshold=np.nan)

def eliminar_columnas_filas(image,umbral):
	columnas = np.mean(image,axis=0)
	filas = np.mean(image,axis=1)

	image = image[:,columnas<umbral]
	return image[filas<umbral,:]
imagenes = listdir("sin_procesar")
n_samples = len(imagenes)
tam = (9,4)
X = np.zeros((n_samples,tam[0]*tam[1]))
Y = np.zeros(n_samples)
images = np.zeros((n_samples,tam[0],tam[1]))

for i in range(n_samples):
	file = imagenes[i]
	image = img.imread(join("procesado",file))

	image = eliminar_columnas_filas(image,0.9)
	#image = 1- image
	
	image = resize(image, tam)
	
	images[i,:,:] = image
	image=np.reshape(image,(1,len(image[0])*len(image))).astype(int)
	image = np.asarray(image[0])
	
	if file[-5] =='A':
		Y[i] = 0
	elif file[-5] == 'B':
		Y[i] = 1
	elif file[-5] == 'C':
		Y[i] = 2
	elif file[-5] == 'D':
		Y[i] = 3
	elif file[-5] == 'E':
		Y[i] = 4
	elif file[-5] == 'F':
		Y[i] = 5
	elif file[-5] == 'G':
		Y[i] = 6
	elif file[-5] == 'H':
		Y[i] = 7
	elif file[-5] == 'I':
		Y[i] = 8
	elif file[-5] == 'J':
		Y[i] = 9

	X[i,:]=image
	#images[i,:,:] = image








#print(X)
#print(y)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.n_classes_=[0,1,2,3,4,5,6,7,8,9]
#clf.n_classes_=[0,1,2,3,4,5,6,7,8,9]


clf.fit(X[:n_samples //2], Y[:n_samples // 2])

predicted = clf.predict(X[n_samples//2:])
expected = Y[n_samples//2:]
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
images_and_predictions = list(zip(images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:25]):
    plt.subplot(5, 5, index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #print(65+prediction)
    plt.title('Prediction: %c\nExpected: %c' % (chr(int(65+prediction)), chr(int(65+expected[index]))))
print("Tasa de acierto %f\n"%(np.sum(predicted==expected)*1./len(predicted)))
plt.show()

