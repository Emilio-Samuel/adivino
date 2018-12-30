import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.transform import resize
image = img.imread('procesado/l00000_A.png')
umbral = 0.9
columnas = np.mean(image,axis=0)
filas = np.mean(image,axis=1)

image = image[:,columnas<umbral]
image = image[filas<umbral,:]
plt.figure()
plt.imshow(image)
plt.gray()
x = np.linspace(9,100,6).astype(int)
y = np.linspace(4,50,6).astype(int)
plt.figure()
print(x,y)
for i in range(6):

  aux = np.copy(image)
  plt.subplot(2,3,i+1)
  plt.gray()
  aux = resize(aux, (x[i],y[i]))
  plt.imshow(aux)
plt.show()
