# Image Processing using Centroidal Voronoi Tesselations

This is a library to process grayscale images using Centroidal Voronoi Tesselations (CVTs). CVTs allow us to partition the image into regions of similar shades. This allows us to reduce the number of shades needed, thus reducing the space needed to store the image. We can also perform image segmentation.

## Shade Reduction

```python
	import matplotlib.pyplot as plt
	import cvt_lib as cvt
	import numpy as np

	# load image
	data = cvt.read_image("images/starfish.png")
	
    # perform a CVT
	generators = np.linspace(0,255,10).tolist()
	generators_new, E, it = cvt.cvt(data, generators, 1e-2, 5)
	
	# render image
	data1 = cvt.cvt_render(data, generators_new)
	
	# show original image
	plt.figure(2)
	plt.subplot(121)
	plt.title("Original image")
	plt.imshow(data, cmap='gray')

	
	# show new image
	plt.subplot(122)
	plt.title("Image with 10 shades")
	plt.imshow(data1, cmap='gray')
```
![Reduced starfish](https://github.com/lukasbystricky/image_processing_CVT/blob/master/images/starfish_reduce.png "Reduced starfish")

## Image Segmentation

```python
	data = cvt.read_image("images/clock.png")
	(sketch, generators) = cvt.image_segmentation(data, 3, 1e-2, 5)
	
	# show original image
	plt.figure(1)
	plt.subplot(121)
	plt.title("Original image")
	plt.imshow(data, cmap='gray')

	
	# show new image
	plt.subplot(122)
	plt.title("Segmented image")
	plt.imshow(sketch, cmap='gray')
```
![Segmented clock](https://github.com/lukasbystricky/image_processing_CVT/blob/master/images/clock_segmented.png "Segmented clock")
# Test-Repositoryu
