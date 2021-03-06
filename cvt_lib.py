from scipy import misc
from operator import truediv
import matplotlib.pyplot as plt
import numpy as np

def read_image(imname):

	# load image, flatten to grayscale if needed
	data = misc.imread(imname, True) 
	
	return data

def histogram(imdata):

	im1d = np.reshape(imdata, (np.product(imdata.shape)),1)
	plt.hist(im1d)
	
	plt.title("Histogram of Shades")
	plt.xlabel("Shade")
	plt.ylabel("Number of pixels")
	
	return 0

def voronoi_zones(imdata, generators):
	
	shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
	zone_data = np.zeros(shape)
	
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
					
			zone_data[i,j] = k_opt			
	
	return zone_data

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

def cvt_step(imdata, generators_old):
	
	shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
	
	bins = [0]*len(generators_old)
	bin_count = [0]*len(generators_old)
	
	#loop over all pixels
	for i in range(0, shape[0]):
		for j in range(0, shape[1]):
			
			min_dist = float("inf")
			k_opt = 0
			
			# loop over generators
			for k in range(len(generators_old)):
				dist = abs(imdata[i,j] - generators_old[k])
				
				if dist < min_dist:
					min_dist = dist
					k_opt = k
					
			bins[k_opt] += imdata[i,j]
			bin_count[k_opt] += 1

	generators_new = map(truediv, bins, bin_count)
					
	return generators_new

def cvt(imdata, generators, tol, max_iter):
	
	#compute initial energy
	E = []
	E.append(compute_energy(imdata, generators))
	
	dE = float("inf")
	it = 0
	
	while (dE > tol and it < max_iter):
		generators = cvt_step(imdata, generators)
		
		
		E.append(compute_energy(imdata, generators))
		it += 1
		dE = abs(E[it] - E[it-1])/E[it]
	
	return (generators, E, it)
	
def compute_energy(imdata, generators):
	
	shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
	energy =  0
	
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
					
			energy += min_dist
					
	return energy
	
def image_segmentation(imdata, n_segments, tol, max_iter):
	
	shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
	sketch = np.ones(shape)
	generators = np.linspace(0,255,n_segments).tolist()
	
	(generators, E, it) = cvt(imdata, generators, tol, max_iter)
	
	zones = voronoi_zones(imdata, generators)
	
	#loop over all pixels, except boundary of image
	for i in range(1, shape[0] - 1):
		for j in range(1, shape[1] - 1):
			
			z = zones[i,j]
			if not (z == zones[i-1,j] and z == zones[i+1,j] and z == zones[i,j-1] and
						z == zones[i,j+1] and z == zones[i-1,j-1] and z == zones[i-1,j+1]
						and z == zones[i+1,j-1] and z == zones[i-1,j+1]):
							sketch[i,j] = 0
	
	return (sketch, generators)
