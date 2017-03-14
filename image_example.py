import cvt_lib as cvt
from scipy import misc
import matplotlib.pyplot as plt

def main():
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

    return 0

main()
    
