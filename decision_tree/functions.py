# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 23:07:50 2018

@author: Jianbing_Dong
"""

#%%
import numpy as np
import pickle
import os

#%%
def isDsame(dataset):
    """
    This function is used to judge if the classes in dataset is all the same.
    """
    class_ = None
    for key, value in dataset.items():
        if key == '属性':
            continue
        class_i =  dataset[key][-1]
        if class_ is None:
            class_ = class_i
        elif class_i != class_:
            return False, None
            
    return True, class_
    
    
def isDsamevalue_in_A(dataset, attributes):
    
    if len(attributes) != 1:
        return False, None
        
    _value = None
    class_ = {}
    index_ = dataset['属性'].index(attributes[0])
    for key in dataset:
        if key == '属性':
            continue
        value_i = dataset[key][index_]
        if _value is None:
            _value = value_i
        elif _value != value_i:
            return False, None
            
        if class_.get(dataset[key][-1]) is None:
            class_[dataset[key][-1]] = 0
        class_[dataset[key][-1]] += 1

    maxclass = None
    maxnum = -1
    for key, value in class_.items():
        if class_[key] > maxnum:
            maxnum = class_[key]
            maxclass = key
            
    return True, maxclass    
        
    
    
    
class Node(object):
    
    def __init__(self):
        """
        This is the node.
        """
        pass
    
    def _set_(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)



def choose_attribute(dataset, attributes):
    maxinforgain = 0
    choosed_attr = None
    for attr in attributes:
       gain_i =  info_gain(dataset, attr)
       if gain_i > maxinforgain:
           maxinforgain = gain_i
           choosed_attr = attr
           
    return choosed_attr



def info_entropy(datasets):
    class_ = {}
    for key, value in datasets.items():
        if key == '属性':
            continue
        if class_.get(datasets[key][-1]) is None:
            class_[datasets[key][-1]] = 0
        class_[datasets[key][-1]] += 1

    Ent = 0
    for key in class_:
        if datasets.get('属性') is not None:
            pk = class_[key] / (len(datasets) - 1)
        else:
            pk = class_[key] / len(datasets)
        Ent -= (pk*np.log2(pk))
        
    return Ent
    
    
def info_gain(datasets, attr):
    gain = info_entropy(datasets)
    index = datasets['属性'].index(attr)
    
    attr_value = {}
    for key, value in datasets.items():
        if key == '属性':
            continue
        if attr_value.get(datasets[key][index]) is None:
            attr_value[datasets[key][index]] = []
        attr_value[datasets[key][index]].append(key)

    for key in attr_value:
        sub_dict_key = get_subdict(datasets, attr_value[key])
        gain -= (len(attr_value[key]) / (len(datasets) - 1)) * info_entropy(sub_dict_key)
            
    return gain
        
        
def get_subdict(dataset, key_list):
    """
    This function is used to get a sub dict form dict, according to 
    the key list.
    """
    outdict = {}
    for key in key_list:
        outdict[key] = dataset[key]

    return outdict


    
def copy_list(original_list):
    newlist = []
    for item in original_list:
        newlist.append(item)
        
    return newlist
    
    
    
def save_tree(root_node, save_name):
    with open(save_name, 'wb') as file:
        pickle.dump(root_node, file, 3)
        print("Save %s successfully!" %save_name)
        
def load_tree(file_name):
    tree = None
    with open(file_name, 'rb') as file:
         tree = pickle.load(file)
         print("Load %s successfully." %file_name)
        
    return tree

#%%


