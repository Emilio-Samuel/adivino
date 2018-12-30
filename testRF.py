"""
Representaremos el logaritmo del area de la imagen
vs tasa de acierto, tiempo trascurrido y la comparacion
del uno contra el otro

Usamos el logaritmo debido a que el area aumenta de forma cuadratica
al variar el tamaño del eje x e y
"""

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
import time
from sklearn import datasets, svm, metrics
#np.set_printoptions(threshold=np.nan)

def eliminar_columnas_filas(image,umbral):
	columnas = np.mean(image,axis=0)
	filas = np.mean(image,axis=1)

	image = image[:,columnas<umbral]
	return image[filas<umbral,:]
def test_random_forest(tam):
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
	

n = 20
x = np.linspace(2,100,n).astype(int)
y = np.linspace(1,58,n).astype(int)
z = zip(x,y)
times = np.zeros(n)
rate = np.zeros(n)
for i,tam in enumerate(z):
	[times[i],rate[i]] = test_random_forest(tam)
plt.figure()
plt.plot(np.log(x*y),times,label="tiempo")
plt.legend(loc='best')
plt.figure()
plt.plot(np.log(x*y),rate,label="tasa de acierto")
plt.legend(loc='best')
plt.figure()
plt.plot(times,rate,label="tasa vs tiempo")
plt.legend(loc='best')
plt.show()