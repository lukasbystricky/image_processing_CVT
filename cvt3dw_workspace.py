import cvt3dw_lib as cvt3d
from scipy import misc
import numpy as np


def main():
    numgen = 2
    numw   = 0
    # load image
    imname = "norm"
    data = cvt3d.read_image("images/" + imname + ".png")
    # perform a CVT on each color array
    randgen = np.random.rand(numgen,3)*256
    
    opt8gen = np.array([[ 75, 75, 75],[ 75, 75,181],\
                        [ 75,181, 75],[ 75,181,181],\
                        [181,181, 75],[181,181,181],\
                        [181, 75, 75],[181, 75,181]])

    subopt8gen = np.array([[  0,  0,  0],[  0,  0,255],\
                           [  0,255,  0],[  0,255,255],\
                           [255,255,  0],[255,255,255],\
                           [255,  0,  0],[255,  0,255]])              
    
    subopt4gen = np.array([[  75,75,  75],[75,75,75],\
                           [  75,  75,75],[75,  75,  75]])  
    
    randgen
    subopt4gen    
    opt8gen
    subopt8gen
    
    generators_new, E, it, weights = cvt3d.cvt(data, subopt4gen, 1e-4, 25, numw)
    

    data1 = cvt3d.cvt_render(data, generators_new, weights, numw)

    savename = imname + "CVT" + str(numgen) + "w" + str(numw)
    misc.imsave("cvt_images/" + savename + "samegens.png", data1)        

    return 0

main()