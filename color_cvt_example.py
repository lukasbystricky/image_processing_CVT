import cvt_lib as cvt
from scipy import misc
import numpy as np

def main():
 
    numgen = 2
    # load image
    data = cvt.read_image("images/starfish.png")
    
    #Spllit image into rgb arrays
    r_data, g_data, b_data = cvt.rgb_split(data)   
    
    # perform a CVT on each color array
    r_generators = np.random.rand(numgen)*250
    r_generators_new, E, it = cvt.cvt(r_data, r_generators, 1e-4, 10)
    r_data1 = cvt.cvt_render(r_data, r_generators_new)
    
    g_generators = np.random.rand(numgen)*250
    g_generators_new, E, it = cvt.cvt(g_data, g_generators, 1e-4, 10)
    g_data1 = cvt.cvt_render(g_data, g_generators_new)
    
    b_generators = np.random.rand(numgen)*250
    b_generators_new, E, it = cvt.cvt(b_data, b_generators, 1e-4, 10)
    b_data1 = cvt.cvt_render(b_data, b_generators_new)

    #Recombine CVT arrays to full image    
    comdata = cvt.rgb_recombine(r_data1,g_data1,b_data1)

    #Save CVT'd images to files
    misc.imsave("cvt_images/CVT_r.png", r_data1)    
    misc.imsave("cvt_images/CVT_g.png", g_data1)   
    misc.imsave("cvt_images/CVT_b.png", b_data1)     
    misc.imsave("cvt_images/CVT_Combine.png", comdata)     
  
    return 0

main()
    