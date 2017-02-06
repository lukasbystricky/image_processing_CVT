import cvt3dw_lib as cvt3d
from scipy import misc
import numpy as np

def main():
    numgen = 8
    numw   = 0
    # load image
    imname = "norm"
    data = cvt3d.read_image("images/" + imname + ".png")
    # perform a CVT on each color array
    generators = np.random.rand(numgen,3)*256

    generators_new, E, it, weights = cvt3d.cvt(data, generators, 1e-4, 5, numw)
    
    data1 = cvt3d.cvt_render(data, generators_new, weights, numw)

    savename = imname + "CVT" + str(numgen) + "w" + str(numw)
    misc.imsave("cvt_images/" + savename + ".png", data1)        

    return 0

main()
    