import numpy as np
import pandas as pd
import scipy.spatial.distance as ds
import time

def main(dataframe):
    start = time.time()
    #Find dimension of data and establish k
    dataframe = pd.DataFrame.to_numpy(dataframe)
    dataset = np.delete(dataframe, 0, 1)
    dim = dataset.shape[1]
    k = 5

    #Find first random centroid for k-means++ centroid allocation
    init_list = list(range(0, 500))
    step_avg = []
    sel_int = [np.random.randint(0, 499)]
    ci = [dataset[sel_int[0]]]

    #Use distance-based probability to select all centroids
    for i in range(k):
        step_avg.append([list(ci[0])])
        init_list.pop(sel_int[0])
        dist_list_dim = ds.cdist(ci, dataset[init_list])**2
        dist_list = dist_list_dim[0] / np.sum(dist_list_dim[0])
        sel_int = np.random.choice(init_list, 1, p=dist_list)
        ci = [dataset[sel_int[0]]]

    #Initialize comparison lists and iterations
    last_avg = np.ones([k, dim])
    comparison = last_avg == step_avg
    iteration = 0
    max_iterations = 50

    #Main K-means loop
    while comparison.all() != True and iteration != max_iterations:
        result_list = list(range(0, 500))
        cycle_list = np.ndarray.tolist(np.copy(step_avg))
        last_avg = np.copy(step_avg)
        squeeze_step = np.squeeze(step_avg)

        #Assign points to their clusters
        j = 0
        for i in dataset:
            val = np.argmin(ds.cdist([i], squeeze_step)[0])
            cycle_list[val].append(list(i))
            result_list[j] = val
            j += 1
        
        #Calculate the new averages
        k = 0
        for i in cycle_list:
            avg = [sum(x)/len(x) for x in zip(*i)]
            step_avg[k] = [avg]
            k += 1
        
        #Update comparison and iteration
        comparison = last_avg == step_avg
        iteration += 1

        #Print an interation update
        #print(iteration, comparison.all(), (np.sum(last_avg)-np.sum(step_avg)))

    #Find elapsed time
    end = time.time()
    print('Kmeans clustering computed in {} seconds'.format(round(end-start, 2)))
    return(result_list)
    

#Run the script from CSV
# data_file = 'csv_data.csv'
# dataframe = pd.read_csv(data_file, skiprows=0)
# results = main(dataframe)
# print(results)    
    


