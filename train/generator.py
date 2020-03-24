import numpy as np
import random
import copy
from utils import convert_data
from PIL import Image
import os
random.seed(42)

def s2s_data_generator(s2s_df=duplets, all_catalog=catalog_images, batch_size=None):
    
    """
    A data generator for generating triplets, i.e, (input_image, positive_image, negative_image) on the fly before training.
    Select a random input image from the training set and select an positive_image with 
    same product id and negative image with different product id and generate batch of triplets
    
    The function keeps yielding a batch of triplets until the whole training process is complete.
    """
    orig_index_list = duplets.index.tolist()
    all_shop_index_list = catalog_images.index.tolist()
    dummy = np.zeros((1, 3 * N))

    while True:
    
        q_list = list()
        p_list = list()
        n_list = list()
        dummy_list = list()
        
        index_list = copy.copy(orig_index_list)
        
        while len(index_list) > 0:

            index = random.choice(index_list)
            product_id = duplets.loc[index, 'product_id']
            
            q_temp = duplets.loc[index, 'street_images']
            q_img = os.path.join(Path, q_temp + '.jpeg')
            
            p_temp = duplets.loc[index, 'shop_images']
            p_img = os.path.join(Path, p_temp + '.jpeg')

            while True:
                idx = random.choice(all_shop_index_list)
                prod_idx = catalog_images.loc[idx, 'product_id']

                if prod_idx != product_id:
                    temp = random.choice(catalog_images.loc[idx, 'shop_images'])
                    n_img = os.path.join(Path, temp + '.jpeg')

            q_img = os.path.join(Path, q_index + '.jpeg')
            p_img = os.path.join(Path, p_index + '.jpeg')
            n_img = os.path.join(Path, n_index + '.jpeg')

            res = bbox_mappings[q_index]

            left = res['left']
            top = res['top']
            right = left + res['width']
            bottom = top + res['height']


            query_img = Image.open(q_img)
            query_crop = query_img.crop((left, top, right, bottom))
            positive_img = Image.open(p_img)
            negative_img = Image.open(n_img)

   
            query = np.array(query_crop.resize((300,300), Image.NEAREST))
            positive = np.array(positive_img.resize((300,300), Image.NEAREST))
            negative = np.array(negative_img.resize((300,300), Image.NEAREST))
                
            
            q_list.append(query_array)
            p_list.append(positive_array)
            n_list.append(negative_array)
            dummy_list.append(dummy)

            
            index_list.remove(index)

            if len(q_list) == batch_size or (len(index_list) == 0 and len(q_list) > 0):
                yield convert_data(q_list, p_list, n_list, dummy_list)
                q_list = list()
                p_list = list()
                n_list = list()
                dummy_list = list()