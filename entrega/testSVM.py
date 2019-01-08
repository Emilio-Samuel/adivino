"""
Representaremos el logaritmo del area de la imagen
vs tasa de acierto, tiempo trascurrido y la comparacion
del uno contra el otro

Usamos el logaritmo debido a que el area aumenta de forma cuadratica
al variar el tamaño del eje x e y
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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
import time
from sklearn import datasets, svm, metrics
#np.set_printoptions(threshold=np.nan)

def eliminar_columnas_filas(image,umbral):
	columnas = np.mean(image,axis=0)
	filas = np.mean(image,axis=1)

	image = image[:,columnas<umbral]
	return image[filas<umbral,:]
def test(tam,rf=True,svc=True):
	imagenes = listdir("sin_procesar")
	n_samples = len(imagenes)

	X = np.zeros((n_samples,tam[0]*tam[1]))
	Y = np.zeros(n_samples)
	for i in range(n_samples):
		file = imagenes[i]
		image = img.imread(join("procesado",file))

		image = eliminar_columnas_filas(image,0.9)
		#image = 1- image
		
		image = resize(image, tam)
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
	sol = []
	sol.append(random_forest(X,Y,n_samples))
	sol.append(svc_classification(X,Y,n_samples))
	sol.append(KNN_test(X,Y,n_samples))
	return sol
def svc_classification(X,Y,n_samples):

	clf = svm.SVC(gamma=0.001)

	

	clf.fit(X[:n_samples //2], Y[:n_samples // 2])
	start_time = time.time()
	predicted = clf.predict(X[n_samples//2:])
	elapsed_time = time.time() - start_time
	
	expected = Y[n_samples//2:]

	return [elapsed_time,sum(expected==predicted)*1./len(expected)]

def random_forest(X,Y,n_samples):
	clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=0)
	clf.n_classes_=[0,1,2,3,4,5,6,7,8,9]

	

	clf.fit(X[:n_samples //2], Y[:n_samples // 2])
	start_time = time.time()
	predicted = clf.predict(X[n_samples//2:])
	elapsed_time = time.time() - start_time
	
	expected = Y[n_samples//2:]

	return [elapsed_time,sum(expected==predicted)*1./len(expected)]
def KNN_test(X,Y,n_samples):
	clf = KNeighborsClassifier(n_neighbors=10)

	

	clf.fit(X[:n_samples //2], Y[:n_samples // 2])
	start_time = time.time()
	predicted = clf.predict(X[n_samples//2:])
	elapsed_time = time.time() - start_time
	
	expected = Y[n_samples//2:]

	return [elapsed_time,sum(expected==predicted)*1./len(expected)]
n = 50
x = np.linspace(4,200,n).astype(int)
y = np.linspace(3,100,n).astype(int)
z = zip(x,y)
timesRF = np.zeros(n)
rateRF = np.zeros(n)
timesSVC = np.zeros(n)
rateSVC = np.zeros(n)
timesKNN = np.zeros(n)
rateKNN = np.zeros(n)
for i,tam in enumerate(z):
	print(i*1./n)
	[[timesRF[i],rateRF[i]],[timesSVC[i],rateSVC[i]],[timesKNN[i],rateKNN[i]]] = test(tam)

plt.figure()
plt.title("Times")
plt.plot(np.log(x*y),timesRF,label="random forest")
plt.plot(np.log(x*y),timesSVC,label="SVC")
plt.plot(np.log(x*y),timesKNN,label="KNN")
plt.legend(loc='best')
plt.figure()
plt.title("tasa de acierto")
plt.plot(np.log(x*y),rateRF,label="random forest")

plt.plot(np.log(x*y),rateSVC,label="SVC")
plt.plot(np.log(x*y),rateKNN,label="KNN")
plt.legend(loc='best')
plt.figure()
plt.title("Tasa vs Tiempo")
plt.plot(timesRF,rateRF,label="random forest")
plt.plot(timesSVC,rateSVC,label="SVC")
plt.plot(timesKNN,rateKNN,label="KNN")
plt.legend(loc='best')
np.savetxt('dataRFKNNSVC_Clasificar',np.stack((timesRF,rateRF,timesSVC,rateSVC,timesKNN,rateKNN)))
plt.show()