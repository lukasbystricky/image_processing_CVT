#!/usr/bin/env python

import matplotlib.pyplot as plt
import cvt_lib as cvt

def main():
 
	# load image
	data = cvt.read_image("images/clock.jpg")

    # perform a CVT
	generators = [5,10,20]
	generators_new, E, it = cvt.cvt(data, generators, 1e-2, 5)
	
	# render image
	data1 = cvt.cvt_render(data, generators_new)
	
	#histogram of colors
	plt.figure(1)
	plt.subplot(211)
	cvt.histogram(data)
	
	# plot energy
	plt.subplot(212)
	plt.plot(E)	
	plt.xlabel("Iteration")
	plt.ylabel("Energy")
		
	plt.show()
		
    # show original image
	plt.figure(2)
	plt.subplot(121)
	plt.imshow(data, cmap='gray')

	
	# show new image
	plt.subplot(122)
	plt.imshow(data1, cmap='gray')
	
	plt.show()

	return 0

main()