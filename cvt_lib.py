from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

def read_image(imname):

	# load image
	data = misc.imread(imname) 
	
	return data

def histogram(imdata):

	im1d = np.reshape(imdata, (np.product(imdata.shape)),1)
	plt.hist(im1d)
	
	plt.title("Histogram of Shades")
	plt.xlabel("Shade")
	plt.ylabel("Number of pixels")
	
	plt.show()
	return 0


