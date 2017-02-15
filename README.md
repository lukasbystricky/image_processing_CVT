# Image Processing using Centroidal Voronoi Tesselations

This is a library to process grayscale images using Centroidal Voronoi Tesselations (CVTs). CVTs allow us to partition the image into regions of similar shades. This allows us to reduce the number of shades needed, thus reducing the space needed to store the image. We can also perform image segmentation. Using a 3D CVT allows individual colors to become shades.

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

## 3D CVT Color Reduction

```python
    numw   = 0  #Weight Exponent (Under Construction)
    
    # load Image
    imname = "starfish"
    data = cvt3d.read_image("images/" + imname + ".png")
    
    # Create initial generators
    randgen1 = np.random.rand(4,3)*256
    randgen2 = np.random.rand(8,3)*256
    randgen3 = np.random.rand(12,3)*256 
      
    # Perform 3D CVT, Render Image, and Save Image 1
    generators_new, E, it, weights = cvt3d.cvt(data, randgen1, 1e-4, 10, numw)    
    data1 = cvt3d.cvt_render(data, generators_new, weights, numw)
    misc.imsave("cvt_images/" + imname + "4.png", data1)  
    
    # Perform 3D CVT, Render Image, and Save Image 2
    generators_new, E, it, weights = cvt3d.cvt(data, randgen2, 1e-4, 10, numw)    
    data2 = cvt3d.cvt_render(data, generators_new, weights, numw)
    misc.imsave("cvt_images/" + imname + "8.png", data2)  

    # Perform 3D CVT, Render Image, and Save Image 3
    generators_new, E, it, weights = cvt3d.cvt(data, randgen3, 1e-4, 10, numw)    
    data3 = cvt3d.cvt_render(data, generators_new, weights, numw)
    misc.imsave("cvt_images/" + imname + "12.png", data3)  

    #Create Plot
    plt.figure(1, figsize=(8, 6))
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(222)
    plt.title("Image with 4 shades")
    plt.imshow(data1)
    
    plt.subplot(223)
    plt.title("Image with 8 shades")
    plt.imshow(data2)
    
    plt.subplot(224)
    plt.title("Image with 12 shades")
    plt.imshow(data3)
    
    plt.savefig("CVTExample.png")
      
    return 0
```
![Cable Example](https://github.com/lukasbystricky/image_processing_CVT/blob/color_cvt/CVTExample.png "Cable Example")

## 3D CVT Color Image Segmentation and Image Averaging

```python
    numw   = 0  #Weight Exponent (Under Construction)
    
    # load Image
    imname = "starfish"
    data = cvt3d.read_image("images/" + imname + ".png")
    
    # Create initial generators
    randgen1 = np.random.rand(4,3)*256
    randgen2 = np.random.rand(8,3)*256
    randgen3 = np.random.rand(12,3)*256 
      
    # Perform 3D CVT, Render Image, and Save Image 1
    generators_new, E, it, weights = cvt3d.cvt(data, randgen1, 1e-4, 10, numw)    
    data1 = cvt3d.cvt_render(data, generators_new, weights, numw)
    misc.imsave("cvt_images/" + imname + "4.png", data1)  
    
    # Perform 3D CVT, Render Image, and Save Image 2
    generators_new, E, it, weights = cvt3d.cvt(data, randgen2, 1e-4, 10, numw)    
    data2 = cvt3d.cvt_render(data, generators_new, weights, numw)
    misc.imsave("cvt_images/" + imname + "8.png", data2)  

    # Perform 3D CVT, Render Image, and Save Image 3
    generators_new, E, it, weights = cvt3d.cvt(data, randgen3, 1e-4, 10, numw)    
    data3 = cvt3d.cvt_render(data, generators_new, weights, numw)
    misc.imsave("cvt_images/" + imname + "12.png", data3)  

    #Create Plot
    plt.figure(1, figsize=(8, 6))
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(222)
    plt.title("Image with 4 shades")
    plt.imshow(data1)
    
    plt.subplot(223)
    plt.title("Image with 8 shades")
    plt.imshow(data2)
    
    plt.subplot(224)
    plt.title("Image with 12 shades")
    plt.imshow(data3)
    
    plt.savefig("CVTExample.png")
      
    return 0
```
![Dog Example](https://github.com/lukasbystricky/image_processing_CVT/blob/color_cvt/CVTSegmentExample.png "Dog Example")
