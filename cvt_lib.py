def read_image(imname):
	from scipy import misc
	
	# load image
	data = misc.imread(imname) 
	
	return data
