from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Lambda, BatchNormalization, Dropout, LeakyReLU
from custom_tf import MaskGlobalAveragePooling1D

def baselineNet():
    input = Input(shape = (None,2))
    x = Conv1D(filters=64,kernel_size=3,padding='valid', activation='relu')(input)
    #x = Conv1D(filters=64,kernel_size=5,padding='valid')(x) 
    x = GlobalAveragePooling1D()(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    
    return model

def baselineMaskNet():
    inp = Input(shape = (None,2))
    mask = Input(shape = (None,1))
    
    x = Conv1D(filters=64,kernel_size=3,padding='valid',activation='relu')(inp)
    m = Lambda(lambda x: x[:,2:,:])(mask) #output_shape=lambda input_shape: (input_shape[0],input_shape[1]-2,input_shape[2])
    # y = Conv1D(filters=64,kernel_size=3,padding='valid')(mask)
    #x = Conv1D(filters=64,kernel_size=3,padding='valid')(x)
    
    x = MaskGlobalAveragePooling1D()(x, input_mask=m)
    
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    m = Model(inputs=[inp, mask], outputs=output)
    
    return m

def maskConvNet():
    inp = Input(shape = (None,2))
    mask = Input(shape = (None,1))
    
    x = Conv1D(filters=32,kernel_size=3,padding='valid', activation='relu')(inp)
    x = Conv1D(filters=64,kernel_size=3,padding='valid', activation='relu')(x)
    x = Conv1D(filters=64,kernel_size=3,padding='valid', activation='relu')(x)
    m = Lambda(lambda x: x[:,4:,:])(mask)
    
    x = MaskGlobalAveragePooling1D()(x, input_mask=m)
    
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    m = Model(inputs=[inp, mask], outputs=output)
    
    return m

def maskDenseNet():
    inp = Input(shape = (None,2))
    mask = Input(shape = (None,1))
    
    x = Conv1D(filters=256,kernel_size=3,padding='valid', activation='relu')(inp)
    m = Lambda(lambda x: x[:,2:,:])(mask)
    
    x = MaskGlobalAveragePooling1D()(x, input_mask=m)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    m = Model(inputs=[inp, mask], outputs=output)
    
    return m

def maskModelDenseReg():
    inp = Input(shape = (None,2))
    mask = Input(shape = (None,1))
    
    x = Conv1D(filters=256,kernel_size=3,padding='valid')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(filters=128,kernel_size=3,padding='valid',activation='relu')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    m = Lambda(lambda x: x[:,4:,:])(mask)
    
    x = MaskGlobalAveragePooling1D()(x, input_mask=m)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.33)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.33)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.33)(x)
    x = Dense(16, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    m = Model(inputs=[inp, mask], outputs=output)
    
    return m