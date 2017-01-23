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

def cvt_render(imdata, generators):

	shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
	imdata_new = np.zeros(shape)
	
	#loop over all pixels
	for i in range(0, shape[0]):
		for j in range(0, shape[1]):
			
			min_dist = float("inf")
			k_opt = 0
			
			# loop over generators
			for k in range(len(generators)):
				dist = abs(imdata[i,j] - generators[k])
				
				if dist < min_dist:
					min_dist = dist
					k_opt = k
					
			imdata_new[i,j] = generators[k_opt]
			
	
	return imdata_new
