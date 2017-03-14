import cvt_lib as cvt
from scipy import misc
import matplotlib.pyplot as plt

def main():   
    #Load Image
    imname = "earth2"
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
      
    return 0

main()
    