import cvt3dw_lib as cvt3d
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def main():
    numw   = 0  #Weight Exponent (Under Construction)
    
    # load Image
    imname = "cables"
    data = cvt3d.read_image("images/" + imname + ".png")
    
    # Create initial generators
    randgen1 = np.random.rand(4,3)*256
    randgen2 = np.random.rand(8,3)*256
    randgen3 = np.random.rand(12,3)*256 

        
    # Perform 3D CVT, Render Image, and Save Image 1
    generators_new, E, it, weights = cvt3d.cvt(data, randgen1, 1e-4, 20, numw)    
    data1 = cvt3d.cvt_render(data, generators_new, weights, numw)
    misc.imsave("cvt_images/" + imname + "4.png", data1)  

    # Perform 3D CVT, Render Image, and Save Image 2
    generators_new, E, it, weights = cvt3d.cvt(data, randgen2, 1e-4, 20, numw)    
    data2 = cvt3d.cvt_render(data, generators_new, weights, numw)
    misc.imsave("cvt_images/" + imname + "8.png", data2)  

    # Perform 3D CVT, Render Image, and Save Image 3
    generators_new, E, it, weights = cvt3d.cvt(data, randgen3, 1e-4, 20, numw)    
    data3 = cvt3d.cvt_render(data, generators_new, weights, numw)
    misc.imsave("cvt_images/" + imname + "12.png", data3)  

    #Create Plot
    plt.figure(1, figsize=(8, 6))
    
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(222)
    plt.title("Image with 4 shades")
    plt.imshow(data1/255)
    
    plt.subplot(223)
    plt.title("Image with 8 shades")
    plt.imshow(data2/255)
    
    plt.subplot(224)
    plt.title("Image with 12 shades")
    plt.imshow(data3/255)

    plt.savefig("CVTExample.png")
      

    return 0

main()
    