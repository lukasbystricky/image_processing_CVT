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
   # print(generators_new)
    #file.close()
    return generators_new, weights_new

def  cvt(imdata, generators, max_iter, numw):
    
    it = 0
    genshape = generators.shape   
    weights = np.zeros(genshape[0]) + 1
    
    misc.imsave("itfolder/iteration0.png", \
        cvt_render(imdata, generators, weights, numw))
    while (it < max_iter):
        generators, weights = cvt_step(imdata, generators, weights, numw, it)
        it += 1

        misc.imsave("itfolder/iteration" + str(it) + ".png", \
            cvt_render(imdata, generators, weights, numw))            
        print("Iteration " + str(it))

    return generators, weights
    
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

def image_segmentation(imdata, generators, max_iter, numw):

    shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
    sketch = np.ones(shape[0:2])
    r, c = shape[0]-1, shape[1]-1 #max row/collumn index
    generators, weights = cvt(imdata, generators, max_iter, numw)

    zones = voronoi_zones(imdata, generators, weights, numw)
    
    #loop over all pixels, except boundary of image
    for i in range(1, r):
        for j in range(1, c):

            z = zones[i,j]
            if not (z == zones[i-1,j]   and z == zones[i+1,j]   and 
                    z == zones[i,j-1]   and z == zones[i,j+1]   and 
                    z == zones[i-1,j-1] and z == zones[i-1,j+1] and
                    z == zones[i+1,j-1] and z == zones[i-1,j+1]):
                sketch[i,j] = 0
    
    #Check 4 corners
    z = zones[0,0]            
    if not (z == zones[1,0] and z == zones[0,1] and z == zones[1,1]):
        sketch[0,0] = 0
        
    z = zones[r,0]
    if not (z == zones[r,1] and z == zones[r-1,1] and z == zones[r-1,0]):
        sketch[r,0] = 0
        
    z = zones[0,c]
    if not (z == zones[0,c-1] and z == zones[1,c-1] and z == zones[1,c]):
        sketch[0,c] = 0

    z = zones[r,c]
    if not (z == zones[r,c-1] and z == zones[r-1,c-1] and z == zones[r-1,c]):
        sketch[r,c] = 0
        
    #Check vertical edges
    for i in range(1,r):
        z = zones[i,0]
        if not (z == zones[i-1,0] and z == zones[i-1,1] and
                z == zones[i,1]   and z == zones[i+1,1] and
                z == zones[i+1,0]):
            sketch[i,0] = 0   
    
        z = zones[i,c]
        if not (z == zones[i-1,c]   and z == zones[i-1,c-1] and
                z == zones[i,c-1]   and z == zones[i+1,c-1] and
                z == zones[i+1,c]):
            sketch[i,c] = 0   
    
    #Check horizontal edges
    for i in range(1,c):
        z = zones[0,i]
        if not (z == zones[0,i-1] and z == zones[1,i-1] and
                z == zones[1,i]   and z == zones[1,i+1] and
                z == zones[0,i+1]):
            sketch[0,i] = 0  

        z = zones[r,i]
        if not (z == zones[r,i-1] and z == zones[r-1,i-1] and
                z == zones[r-1,i] and z == zones[r-1,i+1] and
                z == zones[r,i+1]):
            sketch[r,i] = 0 
        
    return sketch
 
def voronoi_zones(imdata, generators, weights, numw):

    shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
    zone_data = np.zeros(shape[0:2])

    #loop over all pixels
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):

            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(len(generators)):
                dist = (weights[k]**numw) * \
                    norm(imdata[i,j] - generators[k])**2
                    
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k

            zone_data[i,j] = k_opt			

    return zone_data
    
def average_image(imdata):
    shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
    imdata_new = np.ones(shape)
    r, c = shape[0]-1, shape[1]-1 #max row/collumn index

    for i in range(1, r):
        for j in range(1, c):
            nz = np.array([imdata[i-1,j], imdata[i+1,j],  imdata[i,j-1], 
                          imdata[i,j+1], imdata[i-1,j-1],imdata[i-1,j+1],
                          imdata[i+1,j-1], imdata[i-1,j+1]])
            imdata_new[i,j] = np.mean(nz, axis = 0)
    
    #Check 4 corners
    imdata_new[0,0] = np.mean(np.array([imdata[1,0],imdata[0,1],imdata[1,1]]), axis=0)
    imdata_new[r,0] = np.mean(np.array([imdata[r,1],imdata[r-1,1],imdata[r-1,0]]), axis=0)
    imdata_new[0,c] = np.mean(np.array([imdata[0,c-1],imdata[1,c-1],imdata[1,c]]), axis=0)
    imdata_new[r,c] = np.mean(np.array([imdata[r,c-1],imdata[r-1,c-1],imdata[r-1,c]]), axis=0)
    
    #Check vertical edges
    for i in range(1,r):
        nz = np.array([imdata[i-1,0], imdata[i-1,1], imdata[i,1],
                       imdata[i+1,1], imdata[i+1,0]])  
        imdata_new[i,0] = np.mean(nz, axis = 0)

        nz = np.array([imdata[i-1,c], imdata[i-1,c-1], imdata[i,c-1],
                       imdata[i+1,c-1], imdata[i+1,c]])
        imdata_new[i,c] = np.mean(nz, axis = 0)   
    
    #Check horizontal edges
    for i in range(1,c):
        nz = np.array([imdata[0,i-1], imdata[1,i-1], imdata[1,i],
                       imdata[1,i+1], imdata[0,i+1]])
        imdata_new[0,i] = np.mean(nz, axis = 0)  

        nz = np.array([imdata[r,i-1], imdata[r-1,i-1], imdata[r-1,i],
                       imdata[r-1,i+1], imdata[r,i+1]])
        imdata_new[r,i] = np.mean(nz, axis = 0) 
 
    return imdata_new






