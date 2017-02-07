from scipy import misc
import numpy as np
from numpy.linalg import norm

def read_image(imname):
    # load image, flatten to grayscale if needed
    data = misc.imread(imname, False, "RGB") 
    return data

def cvt_render(imdata, generators, weights, numw):

    shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
    imdata_new = np.zeros(shape)
    genshape = generators.shape
    #imfloor = np.zeros(shape)    
    
    #loop over all pixels
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            
            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(genshape[0]):
                dist = (weights[k]**numw) * \
                            norm(imdata[i,j] - generators[k])**2
                
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k
                    
            imdata_new[i,j] = generators[k_opt]
            #imfloor[i,j] = int(imdata[i,j])
    
    return imdata_new#, imfloor

def cvt_step(imdata, generators_old, weights_old, numw, it):#, bin_count_old):
    #file = open('output.txt', 'a')    
    #print(weights_old)
    shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
    genshape = generators_old.shape
    bins = np.zeros([genshape[0],3])
    bin_count = np.zeros(genshape[0])
    weights_new = np.zeros(genshape[0])
    generators_new = np.zeros([genshape[0],3])
    
    
    #loop over all pixels
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            
            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(len(generators_old)):
                dist = (weights_old[k]**numw) * \
                            norm(imdata[i,j] - generators_old[k])**2
                            
                #file.write(str(it) +' '+ str(k) +' '+ str(dist) + '\n')
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k
            #file.write('\n')        
            #print(k_opt)                    
            bins[k_opt] += imdata[i,j]
            bin_count[k_opt] += 1

    #Fixes divide by zero error    
    for elem in range(len(bin_count)):
        if bin_count[elem] == 0:
            bin_count[elem] = 1#imdata.size/3/genshape[0]
            bins[elem] = np.mean(generators_old, axis = 0)
            
    #print(generators_old)
    for i in range(genshape[0]):
        generators_new[i] = bins[i] / bin_count[i] 
        weights_new[i] = bin_count[i]
    #print(weights_new)
    #print(generators_new)
    #file.close()
    #print(weights_new)
    return generators_new, weights_new

def  cvt(imdata, generators, tol, max_iter, numw):
    
    #compute initial energy
    it = 0
    genshape = generators.shape   
    weights = np.zeros(genshape[0]) + imdata.size/3/genshape[0]
    E = []
    #E.append(compute_energy(imdata, generators, weights, numw))
     
    
    dE = float("inf")
    
    
    while (dE > tol and it < max_iter):
        print("Iteration " + str(it + 1)) 
        misc.imsave("itfolder/iteration" + str(it) + ".png", \
                cvt_render(imdata, generators, weights, numw))
        
        generators, weights = cvt_step(imdata, generators, weights, numw, it)
        
        
        #E.append(compute_energy(imdata, generators, weights, numw))
        it += 1
        #dE = abs(E[it] - E[it-1])/E[it]
    
    return generators, E, it, weights
    
def compute_energy(imdata, generators, weights, numw):
    
    imshape  = imdata.shape #shape[0] = # rows, shape[1] = # columns
    genshape = generators.shape
    energy =  0

    #loop over all pixels
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            
            min_dist = float("inf")
            
            # loop over generators)
            for k in range(genshape[0]):
                dist = (weights[k]**numw) * \
                            norm(imdata[i,j] - generators[k])**2
                
                if dist < min_dist:
                    min_dist = dist
                    
            energy += min_dist
                    
    return energy

