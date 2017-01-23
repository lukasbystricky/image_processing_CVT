#!/usr/bin/env python

import matplotlib.pyplot as plt
import cvt_lib as cvt

def main():
 
	# load image
	data = cvt.read_image("clock.jpg")
     
	# change a horizontal bar to a solid color
	# data[50:200,:,:] = [100,60,0]	
	
    # show image
	plt.imshow(data, cmap='gray')
	plt.show()

	return 0

main()
