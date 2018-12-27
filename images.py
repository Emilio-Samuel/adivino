from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.image as img
import scipy.misc
from os import listdir
from skimage import filters
from os.path import isfile, join
from PIL import Image, ImageFilter
from scipy import ndimage
import numpy as np
import cv2



for file in listdir("sin_procesar"):
	image = img.imread(join("sin_procesar",file))
	num_atribs = len(image[0])*len(image)
	break
print("Atributos:",num_atribs)
print("Ejemplos:",len(listdir("sin_procesar")))

dataset = np.zeros((len(listdir("sin_procesar")),num_atribs+1))

i = 0
for file in listdir("sin_procesar"):
	image = img.imread(join("sin_procesar",file))
	val = filters.threshold_otsu(image)
	image =	image>val
	image = ndimage.median_filter(image, 3)
	scipy.misc.imsave(join("procesado",file), image)
	image=np.reshape(image,(1,len(image[0])*len(image))).astype(int)
	image = np.asarray(image[0])
	
	if file[-5] =='A':
		image=np.append(image,0)
	elif file[-5] == 'B':
		image=np.append(image,1)
	elif file[-5] == 'C':
		image=np.append(image,2)
	elif file[-5] == 'D':
		image=np.append(image,3)
	elif file[-5] == 'E':
		image=np.append(image,4)
	elif file[-5] == 'F':
		image=np.append(image,5)
	elif file[-5] == 'G':
		image=np.append(image,6)
	elif file[-5] == 'H':
		image=np.append(image,7)
	elif file[-5] == 'I':
		image=np.append(image,8)
	elif file[-5] == 'J':
		image=np.append(image,9)

	dataset[i,:]=image

	i+=1




X = dataset[:,:-1]
y = dataset[:,-1]
unique, counts = np.unique(y, return_counts=True)
ocurrencias=dict(zip(unique, counts))

print(ocurrencias)



#print(X)
#print(y)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.n_classes_=[0,1,2,3,4,5,6,7,8,9]

clf.fit(X[:np.floor(len(y)*0.9).astype(np.int64),:], y[:np.floor(len(y)*0.9).astype(np.int64)])

a = clf.predict(X[np.floor(len(y)*0.9).astype(np.int64):,:])
aux = a==y[np.floor(len(y)*0.9).astype(np.int64):]
aciertos = np.sum(aux)
tasa_aciertos = aciertos*1./np.floor(len(y)*0.9).astype(np.int64)
print(aciertos)
print(tasa_aciertos)

