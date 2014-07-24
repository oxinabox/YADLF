import numpy as np
import numpyutil as nutil
import itertools as it

def rand_mat(rows,cols=None):
    if cols==None:
        ret =np.random.normal(0,0.01,(rows,)) 
    else:
        ret = np.random.normal(0,0.01,(rows,cols)) 
    return ret

    
def grow_weights(increases, orig_layer_ws):    
    new_ws = []
    for (layer_w,(grow_this,grow_next)) in zip(orig_layer_ws, nutil.pairwise(increases)):
        #add_col(grow_this)
        new_col = rand_mat(layer_w.shape[0],grow_this)
        layer_w = np.append(new_col, layer_w, axis=1)
        #add_row(grow_next)
        new_row = rand_mat(grow_next,layer_w.shape[1])
        layer_w = np.append(new_row, layer_w, axis=0)    
        new_ws.append(layer_w)    
    return new_ws


def grow_upwards_biases(increases, orig_upwards_bs):
    for (next_layer_b,(grow_this,grow_next)) in zip(orig_upwards_bs, nutil.pairwise(increases)):
        new_bias_entries = rand_mat(grow_next,1)
        next_layer_b = np.append(new_bias_entries,next_layer_b)
        yield next_layer_b

def grow_downwards_biases(increases, orig_layer_ws):   
    for (this_layer_b,(grow_this,grow_next)) in zip(orig_layer_ws, nutil.pairwise(increases)):
        new_bias_entries = rand_mat(grow_this,1)
        this_layer_b = np.append(new_bias_entries,this_layer_b)
        yield this_layer_b
        
        
def grow(increases, orig_ws, orig_upwards_bs, orig_downwards_bs=None):
    downward_growth = orig_downwards_bs!=None
    
    new_ws = grow_weights(increases, orig_ws)
    new_upward_bs = list(grow_upwards_biases(increases, orig_upwards_bs))
    
    if not downward_growth:
        return new_ws, new_upward_bs
    
    else:
        new_downward_bs = list(grow_downwards_biases(increases,
                                                     orig_downwards_bs))
        return new_ws, new_upward_bs, new_downward_bs
        
    

def append_top(new_layer_size, orig_ws, orig_upwards_bs, orig_downwards_bs):
    old_top_layer_size = len(orig_upwards_bs[-1])

    new_ws=orig_ws[:]
    new_upward_bs=orig_upwards_bs[:]
    new_downward_bs=orig_downwards_bs[:]
    
    new_ws.append(rand_mat(new_layer_size,old_top_layer_size))
    new_upward_bs.append(rand_mat(new_layer_size))

    
    new_downward_bs.append(rand_mat(old_top_layer_size))
      
    return (new_ws,new_upward_bs, new_downward_bs)
   
def reset_layers(layer_nums_to_reset,*orig):
    #print nutil.arr_info(orig, "orig")
    #print nutil.arr_info(orig[0],"orig[0]")
    assert(len(layer_nums_to_reset)  == len (orig[0]))
    def random_like(mat):
        return rand_mat(*mat.shape)
    
    news = [ [] for _ in orig]
 
    for layer_info in zip(layer_nums_to_reset, *orig):
        reset = layer_info[0]
        knows = layer_info[1:]
        for new_know_list, old_know in zip (news,knows):
            new_know = random_like(old_know) if reset else old_know  
            new_know_list.append(new_know)

    return news

def replace_input_layer(new_size,
                    orig_ws,
                    orig_up_bs,
                    orig_down_bs):

    new_ws=orig_ws[:]
    new_upward_bs=orig_up_bs[:]
    new_downward_bs=orig_down_bs[:]

    sz_next = orig_ws[0].shape[0]
    new_ws[0] = rand_mat(sz_next, new_size)

    new_downward_bs[0] = rand_mat(new_size)        

    return new_ws, new_upward_bs, new_downward_bs

ver = 1.01
