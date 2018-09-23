# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:09:39 2018

@author: Jianbing_Dong
"""


import numpy as np
from functions import *




def createDataset():
    """
    This function is used to create Dataset.
    """
    dataset = {}
    dataset['属性'] = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']
    dataset['1'] = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '是']
    dataset['2'] = ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '是']
    dataset['3'] = ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '是']
    dataset['4'] = ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '是']
    dataset['5'] = ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '是']
    dataset['6'] = ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '是']
    dataset['7'] = ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '是']
    dataset['8'] = ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '是']
    dataset['9'] = ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '否']
    dataset['10'] = ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '否']
    dataset['11'] = ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '否']
    dataset['12'] = ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '否']
    dataset['13'] = ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '否']
    dataset['14'] = ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '否']
    dataset['15'] = ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '否']
    dataset['16'] = ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '否']
    dataset['17'] = ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '否']

    return dataset
    
    
def treeGenerate(dataset, attributes):
    """
    This function is used to create decision tree.
    #arguments:
        dataset: dict, the data used to create tree.
        attributes: list, the attributes used.
    """
    node = Node()
    result = isDsame(dataset)
    if result[0]:
        node._set_(class_=result[1])
        return node
        
    result = isDsamevalue_in_A(dataset, attributes)
    if len(attributes) == 0 or result[0]:
        node._set_(class_=result[1])
        return node
        
    best_attr = choose_attribute(dataset, attributes)
    index = dataset['属性'].index(best_attr)
    node._set_(attr=best_attr, attr_index = index)
    
    in_list = copy_list(attributes)
    in_list.remove(best_attr)
    
    attr_value = {}
    for key, value in dataset.items():
        if key == '属性':
            continue
        if attr_value.get(dataset[key][index]) is None:
            attr_value[dataset[key][index]] = []
        attr_value[dataset[key][index]].append(key)    
        
    node._set_(attr_values={})
    for attr_v in attr_value:
        
        sub_dict_attr_v = get_subdict(dataset, attr_value[attr_v])
        
        if len(sub_dict_attr_v) == 0:
            subnode = Node()
            class_ = {}
            for key in dataset:
                if key == '属性':
                    continue
                if class_.get(dataset[key][-1]) is None:
                    class_[dataset[key][-1]] = 0
                class_[dataset[key][-1]] += 1
            maxnum = 0
            maxclass = None
            for key in class_:
                if class_[key] > maxnum:
                    maxnum = class_[key]
                    maxclass = key
            subnode._set_(class_=maxclass)       
            
        else:
            sub_dict_attr_v['属性'] = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']
            subnode = treeGenerate(sub_dict_attr_v, in_list) 
 
        if node.attr_values.get(attr_v) is None:
            node.attr_values[attr_v] = subnode 
            
    return node
            
                    
    
def decide_with_tree(root_node, data):
    
    decision_result = None
    if hasattr(root_node, 'class_'):
        decision_result = root_node.class_
        return decision_result
        
    else:
        attr_index = root_node.attr_index
        data_attr = data[attr_index]
        attr_node = root_node.attr_values[data_attr]

        in_data = copy_list(data)
        decision_result = decide_with_tree(attr_node, in_data)     
        
    return decision_result



    
    
if __name__ == '__main__':
    
    if not os.path.exists(r'./decision_tree.bin'):
        
        data = createDataset()
        attributes = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    
        decision_tree = treeGenerate(data, attributes)
        
        save_tree(decision_tree, r'./decision_tree.bin')
    else:
        
        decision_tree = load_tree(r'./decision_tree.bin')
    
    test_data = ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.1, 0.8]

    test_result = decide_with_tree(decision_tree, test_data)
    
    print("是否是好瓜：%s" %test_result)
    

    