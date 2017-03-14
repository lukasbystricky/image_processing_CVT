# Image Processing using Centroidal Voronoi Tesselations

This is a library to process images using Centroidal Voronoi Tesselations (CVTs). CVTs allow us to partition the image into regions of similar shades. This allows us to reduce the number of shades needed, thus reducing the space needed to store the image. We can also average images to reduce noise, and perform image segmentation as a method of edge detection. Additionally, a set of images with each containing part, but not the entierty of an image can be combined with a multichannel CVT.

## CVT Color Reduction

The CVT algorithm sorts the RGB value of each pixel in an image according to the generator to which it is closest. The geometric centroid of these 3D points in RGB color space is calculated, with the result being used as generators for the next iteration. Over many iterations the total energy decreases, and the resulting image converges.

In the event that a generator has no pixels assigned to its cluster, the generator is changed to a random color, and an extra CVT iteration is performed. This is of particular importance in weighted CVTs, discussed below.

```python
    # load Image
    imname = "cables"
    data = cvt.read_image("images/" + imname + ".png")
    
    # Create initial generators
    optgen1 = cvt.plusplus(data, 4)
    optgen2 = cvt.plusplus(data, 8)
    optgen3 = cvt.plusplus(data, 12)
    
    # Perform 3D CVT, Render Image, and Save Image 1
    generators_new, weights, E = cvt.cvt(data, optgen1, 1e-4, 20, 0)    
    data1 = cvt.cvt_render(data, generators_new, weights, 0)
    misc.imsave("cvt_images/" + imname + "4.png", data1)  

    # Perform 3D CVT, Render Image, and Save Image 2
    generators_new, weights, E = cvt.cvt(data, optgen2, 1e-4, 20, 0)    
    data2 = cvt.cvt_render(data, generators_new, weights, 0)
    misc.imsave("cvt_images/" + imname + "8.png", data2)  

    # Perform 3D CVT, Render Image, and Save Image 3
    generators_new, weights, E = cvt.cvt(data, optgen3, 1e-4, 20, 0)    
    data3 = cvt.cvt_render(data, generators_new, weights, 0)
    misc.imsave("cvt_images/" + imname + "12.png", data3)  

    #Create Plot
    plt.figure(1, figsize=(8, 6))
    
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(222)
    plt.title("Image with 4 shades")
    plt.imshow(data1/256)
    
    plt.subplot(223)
    plt.title("Image with 8 shades")
    plt.imshow(data2/256)
    
    plt.subplot(224)
    plt.title("Image with 12 shades")
    plt.imshow(data3/256)

    plt.savefig("CVTExample.png")
```
![Cable Example](https://github.com/lukasbystricky/image_processing_CVT/blob/color_cvt/cvt_images/CVTExample.png "Cable Example")

## Choosing Initial Generators with kmeans++

In an ordinary CVT, initial generators are chosen randomly from 3D color space. This can result in an unideal rest position, as well as spending iterations to get to a reasonable position. The kmeans++ algorithm chooses initial points one at a time from existing set of colors. Each generator after the first is chosen randomly, with a probability proportional to its distance from existing points. This causes the initial generators to more accurately represent the image, which means fewer iterations are neccesary to reach a rest position. That being said, the selection process still involves randomness, and may not result in optimal generators every time. 

```python
    # load Image
    imname = "starfish"
    data = cvt.read_image("images/" + imname + ".png")
    
    # Create initial generators
    randgen = np.random.rand(4,3)*256
    optgen = cvt.plusplus(data,4)    
    
    # Perform 3D CVT, Render Image, and Save Image 1
    generators_new, weights, E1 = cvt.cvt(data, randgen, 0, 3, 0)    
    data1 = cvt.cvt_render(data, generators_new, weights, 0)
    misc.imsave("cvt_images/" + imname + "_rand.png", data1)  

    # Perform 3D CVT, Render Image, and Save Image 2
    generators_new, weights, E2 = cvt.cvt(data, optgen, 0, 3, 0)    
    data2 = cvt.cvt_render(data, generators_new, weights, 0)
    misc.imsave("cvt_images/" + imname + "_plus.png", data2)  

    #Create Plot
    plt.figure(1, figsize=(8, 6.5))
    
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(222)
    plt.title("Energy over Iteration Plot")
    plt.plot(E1, color = 'r')
    plt.plot(E2, color = 'b')
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    
    plt.subplot(223)
    plt.title("Random Initial Generators")
    plt.imshow(data1/256)
    
    plt.subplot(224)
    plt.title("kmeans++ Initial Generators")
    plt.imshow(data2/256)

    plt.savefig("CVTExample.png")
```

![Starfish Example](https://github.com/lukasbystricky/image_processing_CVT/blob/origin/color_cvt/cvt_images/CVTppExample.png "Starfish Example")

## Weighted CVT Color Reduction

Adding weights to generators influences how pixels are assigned to each. Typically the weights are dependent on some factor of the generator clusters themselves. In this case, the weight for each generator is the number of pixels assigned to it in the previous iteration, raised to a specific power. For negative exponents, this has the consequence of giving more prominent colors more influence, causing large colors to grow in size and small colors to shrink. For positive exponents, the reverse is true: small colors will grow while large colors shrink, in effect causing each color to have a similar size after repeated iterations. Larger exponents cause these effects to become more pronounced.

Weighted CVTs, particularly those with negative exponents, will frequently cause colors to disapear entierly. This in turn causes the color to be weighted extremely highly, as it contains very few pixels. As a result, the absent color often dominates the image in the next iteration, causing the CVT image to be a solid color. To counter this, whenever a color is absent, and unweighted CVT is performed to ensure each color has an accurately representative number of pixels for weights.

In the below example, the positive averaging weight has more equally distributed colors, where the negative averaging weight has one color covering a small area.

```python
    # load Image
    imname = "zebra"
    data = cvt.read_image("images/" + imname + ".png")
    
    # Create initial generators
    optgen = cvt.plusplus(data,3)
    
    # Perform CVT with Large Negative Weight
    generators_new, weights, E = cvt.cvt(data, optgen, 1e-3, 15, 0)    
    data1 = cvt.cvt_render(data, generators_new, weights, 0)
    misc.imsave("cvt_images/" + imname + "0.png", data1)  

    # Perform CVT with large, negative weight
    generators_new, weights, E = cvt.cvt(data, optgen, 1e-3, 15, -2)    
    data2 = cvt.cvt_render(data, generators_new, weights, -2)
    misc.imsave("cvt_images/" + imname + "-2.png", data2)  

    # Perform CVT with large, positive weight
    generators_new, weights, E = cvt.cvt(data, optgen, 1e-3, 15, 2)    
    data3 = cvt.cvt_render(data, generators_new, weights, 2)
    misc.imsave("cvt_images/" + imname + "2.png", data3)  

    #Create Plot
    plt.figure(1, figsize=(8, 6))
    
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(222)
    plt.title("Image with No Average")
    plt.imshow(data1/256)
    
    plt.subplot(223)
    plt.title("Image with Weight -2")
    plt.imshow(data2/256)
    
    plt.subplot(224)
    plt.title("Image with Weight +2")
    plt.imshow(data3/256)

    plt.savefig("WeightedCVTExample.png")

    return 0
```
![Zebra Example](https://github.com/lukasbystricky/image_processing_CVT/blob/color_cvt/cvt_images/WeightedCVTExample.png "Zebra Example")

## CVT Image Segmentation and Image Averaging

Using a CVT image with a resonably small number of colors allows for easy edge detection. Any pixel that is not entirely surrounded by pixels belonging to the same generator is colored. This essentially draws lines along the borders of each spatial CVT region, marking its edges. 

Image averaging assists in this, effectively "smoothing" out rough edges by removing noise in the original image. This is done by replacing each pixel in the image with the average of the pixels in a surrounding radius, weighted in a way that prioritizes closer pixels. Two possible functions are provided within cvt_lib.py, a Gaussian average and one more computationally efficent. The two parameters, sigma and beta, influence the intensity and radius of the average respectively. The radius should be calculated differently based on the averaging function used, with the equations provided in the comments of the source code. As a default, the second is used. 

```python
    #Load Image
    imname = "smalldog"
    data = cvt.read_image("images/" + imname + ".png")
    data3 = data2 = data1 = data
    
    # Create initial generators
    optgen = cvt.plusplus(data,3)
        
    # Perform CVT with no average
    sketch1, generators, weights, E = \
              cvt.image_segmentation(data1, optgen, 1e-4, 20, 0)
    misc.imsave("cvt_images/" + imname + "0.png", sketch1)  
   
    # Perform CVT with mild average
    data2 = cvt.smoothing_avg(data2, 1.5, 0.5)  
    sketch2, generators, weights, E = \
              cvt.image_segmentation(data2, optgen, 1e-4, 20, 0)
    misc.imsave("cvt_images/" + imname + ".5.png", sketch2)  

    # Perform CVT with severe average
    data3 = cvt.smoothing_avg(data3, 1.5, 0.05)  
    sketch3, generators, weights, E = \
              cvt.image_segmentation(data3, optgen, 1e-4, 20, 0)
    misc.imsave("cvt_images/" + imname + ".05.png", sketch3)  

    #Create Plot
    plt.figure(1, figsize=(8, 6))
    
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(222)
    plt.title("Segmented Image with No Average")
    plt.imshow(sketch1, cmap='gray')
    
    plt.subplot(223)
    plt.title("Segmented Image with Mild Average")
    plt.imshow(sketch2, cmap='gray')
    
    plt.subplot(224)
    plt.title("Segmented Image with Severe Average")
    plt.imshow(sketch3, cmap='gray')

    plt.savefig("CVTSegmentExample.png")
```
![Dog Example](https://github.com/lukasbystricky/image_processing_CVT/blob/color_cvt/cvt_images/CVTSegmentExample.png "Dog Example")

## Multichannel CVT Reconstruction

Multiple images, each depicting a part of the same image, can be recombined using a CVT. This is done through the ordinary CVT process, with the exception of summing the distances of the analogous pixels of each image to determine the assigned generator. This causes colors present in only a single image to be visible on the reconstructed image, essentially averaging each image used. While not currently present, this process can be combined with weights, segmentation, and averaging to further create an optimal CVT image.

```python
    #load Images
    data1 = cvt.read_image("images/thisisa.png")
    data2 = cvt.read_image("images/multichannel.png")
    data3 = cvt.read_image("images/cvt.png")
    
    #Create initial generators
    randgen = np.random.rand(2,3)*256 
    
    #Add image data to array
    data_arr = np.array([data1,data2,data3])
    
    #Perform Multichannel CVT and return combined image
    sketch = cvt.multi_channel(data_arr, randgen, 10)
    
    #Save Combined CVT
    misc.imsave("cvt_images/combined_image.png", sketch)  

    #Create Plot
    plt.figure(1, figsize=(8, 6))
    
    plt.subplot(221)
    plt.title("Image 1")
    plt.imshow(data1)

    plt.subplot(222)
    plt.title("Image 2")
    plt.imshow(data2)
    
    plt.subplot(223)
    plt.title("Image 3")
    plt.imshow(data3)
    
    plt.subplot(224)
    plt.title("Combined Image")
    plt.imshow(sketch/255)

    plt.savefig("mcCVTExample.png")
```
![Text Example](https://github.com/lukasbystricky/image_processing_CVT/blob/color_cvt/cvt_images/mcCVTExample.png "Text Example")
