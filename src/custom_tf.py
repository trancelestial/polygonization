import numpy as np
import copy
import tensorflow as tf
from tensorflow.keras import backend

class MaskGlobalAveragePooling1D(tf.keras.layers.Layer):
    def __init__(self):
        super(MaskGlobalAveragePooling1D, self).__init__()
    
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        
        return tf.TensorShape([input_shape[0], input_shape[2]])
    
    def call(self, input, input_mask):
        mask = input_mask
        mask = tf.cast(mask, backend.floatx())
        
        input *= mask
        
        return backend.sum(input, axis=1) / backend.sum(mask, axis=1)
        #return backend.mean(input, axis=1)

class DataGenerator(tf.keras.utils.Sequence):
    'data generator for feeding into memory parts of dataset'
    def __init__(self, dX, dY, batch_size=32, pc_first_kernel_size=None, pad_zeros=True, mask_model=False):
        'Initialization'
        self.batch_size = batch_size
        self.dX = copy.deepcopy(dX)
        self.dY = copy.deepcopy(dY)
        self.cyclic_pad = pc_first_kernel_size-1 if pc_first_kernel_size != None else pc_first_kernel_size
        self.zero_pad = pad_zeros
        self.mask_model = mask_model

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dX) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        X, Y = self.__data_generation(index)

        return X, Y
   
    # extend with creating mask at the same time
    def pad_zeros(self, xb, longest):
        mask = []
        for i in range(len(xb)):
            l = len(xb[i])
            xb[i].extend([[0, 0]]*(longest-l))
            mask.append([[1]]*l)
            mask[i].extend([[0]]*(longest-l))

        return mask#, diffs
    
    # shortest poly has 4 edges, and first filter is 5
    def pad_cyclic(self, xb):
        for i in range(len(xb)):
            xb[i].extend(xb[i][:self.cyclic_pad])

    def __data_generation(self, index):
        # X : (n_samples, *dim, n_channels)
        xb = copy.deepcopy(self.dX[index*self.batch_size:(index+1)*self.batch_size])
        yb = copy.deepcopy(self.dY[index*self.batch_size:(index+1)*self.batch_size])
        if self.cyclic_pad != None:
            self.pad_cyclic(xb)
        if self.zero_pad:
            longest = max(list(map(len, xb)))
            # print(f'Padding to size: {longest}')
            mask = self.pad_zeros(xb, longest)
            
        if self.mask_model:
            # return np.array(xb), np.array(yb)
            return [np.array(xb), np.array(mask)], np.array(yb)
        else:
            return np.array(xb), np.array(yb)

class PredictDataGenerator(tf.keras.utils.Sequence):
    'data generator for feeding into memory parts of dataset'
    def __init__(self, dX, pad_cyclic = None):
        'Initialization'
        #self.dX = copy.deepcopy(dX)
        self.dX = dX
        self.cyclic_pad = pad_cyclic

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dX)

    def __getitem__(self, index):
        'Generate one batch of data'
        X = self.__data_generation(index)

        return X

    def pad_cyclic(self, xb):
        for i in range(len(xb)):
            xb[i].extend(xb[i][:self.cyclic_pad])

    def __data_generation(self, index):
        # X : (n_samples, *dim, n_channels)
        if index % 300000 == 0:
            print(f'index: {index}')
        xb = self.dX[index]
        
        if self.cyclic_pad != None:
            self.pad_cyclic(xb)
        test = np.expand_dims(np.array(xb),0)
        mask = np.ones((1,test.shape[1],1))

        return [test, mask]