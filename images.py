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
	val = filters.threshold_otsu(image)
	image =	image>val
	image = ndimage.median_filter(image, 3)
	scipy.misc.imsave(join("procesado",file), image)