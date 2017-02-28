import cvt_lib as cvt
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def main():
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

    return 0

main()
    
