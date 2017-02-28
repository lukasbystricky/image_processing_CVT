import cvt_lib as cvt
from scipy import misc
import numpy as np

def main(numgen, numw, sigma, beta):   
    
    #Load Image
    imname = "starfish"
    data = cvt.read_image("images/" + imname + ".png")
    
    #Create Initial Sample Points
    randgen = np.random.rand(numgen,3)*256
    
    #Smooth the Image to ReduceNnoise
    if sigma:   #If smoothing parameter is 0, do not average
        data = cvt.smoothing_avg(data, sigma, beta)
        misc.imsave("Im_Avg/Avg" + imname + \
                    "_S" + str(sigma)  + "_B" + str(beta) + ".png", data)

    #Perform Image Segmentation and CVT
    sketch, generators_new, weights = \
                cvt.image_segmentation(data, randgen, 1e-3, 10, numw)

    #Create CVT Color Image
    cvt_data = cvt.cvt_render(data, generators_new, weights, numw)
    
    #Create Save Name Based on Given Parameters
    savename = imname + "_G" + str(numgen) + "_W" + str(numw) \
                      + "_S" + str(sigma)  + "_B" + str(beta)          
    #Save Images
    misc.imsave("Im_CVT/CVT" + savename + ".png", cvt_data)        
    misc.imsave("Im_Seg/Seg" + savename + ".png", sketch)        

    return 0

main(8,0,1.5,0.05)
