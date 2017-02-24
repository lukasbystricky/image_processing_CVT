import cvt3dw_lib as cvt3d
from scipy import misc
import numpy as np

def main(numgen, numw, sigma, beta):   
    
    #Load Image
    imname = "quinn"
    data = cvt3d.read_image("images/" + imname + ".png")
    
    #Create Initial Sample Points
    randgen = np.random.rand(numgen,3)*256
    
    #Smooth the Image to ReduceNnoise
    if sigma:#If smoothing parameter is 0, do not average
        data = cvt3d.smoothing_avg(data, sigma, beta)

    #Perform Image Segmentation and CVT
    sketch, generators_new, weights = \
                cvt3d.image_segmentation(data, randgen, 1e-3, 10, numw)

    #Create CVT Color Image
    cvt_data = cvt3d.cvt_render(data, generators_new, weights, numw)
    
    #Create Save Name
    savename = imname + "_G" + str(numgen) + "_W" + str(numw) \
                      + "_S" + str(sigma)  + "_B" + str(beta)          
    #Save Images
    misc.imsave("Im_Avg/Avg" + imname + \
                    "_S" + str(sigma)  + "_B" + str(beta) + ".png", data)
    misc.imsave("Im_CVT/CVT" + savename + ".png", cvt_data)        
    misc.imsave("Im_Seg/Seg" + savename + ".png", sketch)        

    return 0

main(6,0,1,0.2)