import cvt_lib as cvt
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def main():
    # load Image
    imname = "zebra"
    data = cvt.read_image("images/" + imname + ".png")
    
    # Create initial generators
    randgen = np.random.rand(3,3)*256 
    
    # Perform CVT with Large Negative Weight
    generators_new, weights = cvt.cvt(data, randgen, 1e-3, 15, 0)    
    data1 = cvt.cvt_render(data, generators_new, weights, 0)
    misc.imsave("cvt_images/" + imname + "0.png", data1)  

    # Perform CVT with large, negative weight
    generators_new, weights = cvt.cvt(data, randgen, 1e-3, 15, -2)    
    data2 = cvt.cvt_render(data, generators_new, weights, -2)
    misc.imsave("cvt_images/" + imname + "-2.png", data2)  

    # Perform CVT with large, positive weight
    generators_new, weights = cvt.cvt(data, randgen, 1e-3, 15, 2)    
    data3 = cvt.cvt_render(data, generators_new, weights, 2)
    misc.imsave("cvt_images/" + imname + "2.png", data3)  

    #Create Plot
    plt.figure(1, figsize=(8, 6))
    
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(222)
    plt.title("Image with No Average")
    plt.imshow(data1/255)
    
    plt.subplot(223)
    plt.title("Image with Weight -2")
    plt.imshow(data2/255)
    
    plt.subplot(224)
    plt.title("Image with Weight +2")
    plt.imshow(data3/255)

    plt.savefig("WeightedCVTExample.png")

    return 0

main()
    