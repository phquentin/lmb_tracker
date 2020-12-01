import numpy as np
from scipy.special import logsumexp

def gibbs_sampler(C, hyp_weights, NUM_SAMPLES, MAX_INVALID_SAMPLES):
    """Computes the weights for each track assignment hypothesis

    Samples the most likley valid assignment mappings for given tracks and measurements based on 
    the corresponding negative log likelihood assignments of C. Based on the sampled mappings it 
    accumulates the corresponding hypothesis weights in hyp_weights which can subsequently be used 
    to compute the existence probability of a track.

    TODO: 
    - Replace lists through predefined arrays
    - Parallelize the sampling
   
    The Algorithm is based on: 
    - Olofsson, J., Veibäck, C., & Hendeby, G. (2017). Sea ice tracking with a spatially indexed labeled 
    multi-Bernoulli filter. In 20th International Conference on Information Fusion (FUSION). Xi’an, China. 

    - Reuter, A. Danzer, M. Stübler, A. Scheel and K. Granström, "A fast implementation of the Labeled 
    Multi-Bernoulli filter using gibbs sampling," 2017 IEEE Intelligent Vehicles Symposium (IV), Los Angeles, 
    CA, 2017, pp. 765-772, doi: 10.1109/IVS.2017.7995809.

    Parameters
    ----------
    C: ndarray
        Cost matrix which contains the negative log likelihood for a track assignment. A track can be assigned to 
        a measurement, to a missed detection or to the death state. 
        Dimension: [Tracks, Measurements + Missed Detection + Death]

    hyp_weights: ndarray
        Accumulated hypothesis weights based on the sampled mappings.
        Dimension: [Tracks, Measurements + Missed Detection + Death]

    NUM_SAMPLES: int
        Number of samples the Gibbs sampler takes from the cost matrix to create the mappings.

    MAX_INVALID_SAMPLES: int
        Maximum number of consecutive invalid samples that do not contain a valid assignment after that the gibbs sampler terminates
        even though the number of NUM_SAMPLES may not be reached.
    """

    assignment_maps = []
    consecutive_invalid_samples = 0 
    NUM_TRACKS = C.shape[0]
    NUM_POS_ASSIGNS = C.shape[1]

    for j in range(NUM_SAMPLES):

        association_map_found = False
        map_nll = 0
    
        while(association_map_found == False):
            association_indices = []
            sample = C * np.random.rand(NUM_TRACKS, NUM_POS_ASSIGNS) 
            # Select a assigment by categorial sampling and check if it is a valid assigment 
            for i in range(NUM_TRACKS):
                min_value_index = np.argmin(sample[i])
                if(min_value_index == NUM_POS_ASSIGNS - 1 or min_value_index == NUM_POS_ASSIGNS - 2 or min_value_index not in association_indices):
                    association_indices.append(min_value_index)
                else:
                    break
            if(len(association_indices) == NUM_TRACKS):
                association_map_found = True

        # Check if the current assignment mapping has already been found, if not apped to assigment maps
        if(association_indices not in assignment_maps):
            assignment_maps.append(association_indices)
            consecutive_invalid_samples = 0

            # Calculate the likelihood of the current assignment mapping
            for row, column in enumerate(association_indices):
                map_nll += C[row,column]

            # Add the likelihood of the current assignment mapping to all corresponding track assignments
            for row, column in enumerate(association_indices):
                hyp_weights[row, column] += map_nll

        else:
            consecutive_invalid_samples += 1

        if(consecutive_invalid_samples >= MAX_INVALID_SAMPLES):
            break

    # Normalization of the accumulated hypothesis weights with broadcasting
    nll_hyp_sum = logsumexp(hyp_weights, axis = -1)
    hyp_weights -= nll_hyp_sum[:,None]
    

    
   








