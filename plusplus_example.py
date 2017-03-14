import cvt_lib as cvt
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def main():
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

    plt.savefig("CVTppExample.png")

    return 0

main()
    
