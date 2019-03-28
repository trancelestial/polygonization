# make this in batch generation
m = np.ones(B,T)
m[0,-2:] = 0
m[2,-1:0] = 0

# create model
inp = Input((B,T,F))(x) # B, T, F
mask = Input(B,T)(m) # B, T
...
mask = tf.expand_dims... # this can be omitted cuz globalAvg already has it implemented. Purpose is to enable broadcasting
x = GlobalAveragePooling1D(...,mask=mask)(x) # in global average pool ine with int_something or something_int change to tf.shape/np.shape
output = Dense()(x)
