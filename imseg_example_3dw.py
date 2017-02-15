import cvt3dw_lib as cvt3d
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def main():
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

    for i in range(1):
        data2 = cvt3d.average_image(data2)  

    # Perform 3D CVT, Render Image, and Save Image 2
    sketch2 = cvt3d.image_segmentation(data2, randgen, 20, numw)
    misc.imsave("cvt_images/" + imname + "1.png", sketch2)  

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
      
    return 0

main()
    