# Image Processing using Centroidal Voronoi Tesselations

This is a library to process images using Centroidal Voronoi Tesselations (CVTs). CVTs allow us to partition the image into regions of similar shades. This allows us to reduce the number of shades needed, thus reducing the space needed to store the image. We can also average images to reduce noise, and perform image segmentation as a method of edge detection

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
```
![Cable Example](https://github.com/lukasbystricky/image_processing_CVT/blob/color_cvt/CVTExample.png "Cable Example")

## 3D CVT Color Image Segmentation and Image Averaging

```python
    numw   = 0  #Weight Exponent (Under Construction)
    
    # load Image
    imname = "smalldog"
    data = cvt3d.read_image("images/" + imname + ".png")
    data3 = data2 = data1 = data
    
    # Create initial generators
    randgen = np.random.rand(2,3)*256
        
    # Perform 3D CVT, Render Image, and Save Image 1
    sketch1 = cvt3d.image_segmentation(data1, randgen, 20, numw)
    misc.imsave("cvt_images/" + imname + "0.png", sketch1)  

    #Perform One Average
    for i in range(1):
        data2 = cvt3d.average_image(data2)  

    # Perform 3D CVT, Render Image, and Save Image 2
    sketch2 = cvt3d.image_segmentation(data2, randgen, 20, numw)
    misc.imsave("cvt_images/" + imname + "1.png", sketch2)  

    #Perform Ten Averages
    for i in range(10):
        data3 = cvt3d.average_image(data3)  

    # Perform 3D CVT, Render Image, and Save Image 3
    sketch3 = cvt3d.image_segmentation(data3, randgen, 20, numw)
    misc.imsave("cvt_images/" + imname + "10.png", sketch3)  

    #Create Plot
    plt.figure(1, figsize=(8, 6))
    
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(222)
    plt.title("Segmented Image with 0 Averages")
    plt.imshow(sketch1, cmap='gray')
    
    plt.subplot(223)
    plt.title("Segmented Image with 1 Average")
    plt.imshow(sketch2, cmap='gray')
    
    plt.subplot(224)
    plt.title("Segmented Image with 10 shades")
    plt.imshow(sketch3, cmap='gray')

    plt.savefig("CVTSegmentExample.png")
```
![Dog Example](https://github.com/lukasbystricky/image_processing_CVT/blob/color_cvt/CVTSegmentExample.png "Dog Example")
