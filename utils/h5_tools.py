import numpy as np
import h5py

def read_h5_data(hf_file,keys=None):
    '''
    read data from h5 file given keys, if keys is None, then return all values
    inputs
    - hf_file
    - keys
    returns
    - values: when given keys,it is list corresponding to keys, otherwise it is dict value = values[key]
    '''
    
    with h5py.File(hf_file,'r') as hf:
        print hf.keys()
        if keys is None:
            values = {} 
            for key in hf.keys():
               values[key] = hf[key][:]
            return values 
        values = []
        for key in keys:
            
            values.append(hf[key][:])
        return values
def read_h5_data_to_dict(hf_file,keys=None):
    '''
    read data from h5 file given keys, if keys is None, then return all values
    inputs
    - hf_file
    - keys
    returns
    - key-values: when given keys,it is list corresponding to keys, otherwise it is dict value = values[key]
    '''
     
    with h5py.File(hf_file,'r') as hf:
        #hf.keys()
        if keys is None:
            values = {} 
            for key in hf.keys():
               values[key] = hf[key][:]
            return values 
        values = {}
        for key in keys:
            
            values[key] = hf[key][:]
        return values


def write_h5_data(hf_file,dict):
    '''
    write dict to hf_file
    inputs
    - dict: key-value pairs
    '''
    with h5py.File(hf_file,'w') as hf:
        hf_dict = {}
        for key, value in dict.iteritems():
            hf_dict[key] = hf.create_dataset(key, value.shape,value.dtype)
            hf_dict[key][...] = value


