from keras.applications import vgg16
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.regularizers import activity_l1,l1
import h5py


# This is the original preset model configuration for 
# replacekinect project
def original(loadweights,weightfile,stop_summary):
    # Vgg model without fully connected layer
    vggmodel = vgg16.VGG16(include_top=False)
    # create fully connected layer
    fc_input = Input(shape=(512,5,10))
    x = Flatten(name='flatten')(fc_input)
    x = Dense(1024, activation='relu',name='fc1')(x)
    x = Dense(1024, activation='relu',name='fc2')(x)
    x = Dense(60,activation='linear',name='predictions')(x)
    fcmodel = Model(fc_input,x)
    # Load the model weights if instructed
    if loadweights:
        fcmodel.load_weights(weightfile)
    # Compile and print summary
    fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
        metrics=['accuracy'])
    if not stop_summary:
        print 'Convolutional Part:'
        vggmodel.summary()
        print 'Fully Connected Part:'
        fcmodel.summary()
    print 'Model loaded'
    return vggmodel, fcmodel

# The number of fully connected layer is doubled, as well as
# the number of neurons per layer
def doubledense(loadweights,weightfile,stop_summary):
    # Vgg model without fully connected layer
    vggmodel = vgg16.VGG16(include_top=False)
    # create fully connected layer
    fc_input = Input(shape=(512,5,10))
    x = Flatten(name='flatten')(fc_input)
    x = Dense(2048, activation='relu',name='fc1')(x)
    x = Dense(2048, activation='relu',name='fc2')(x)
    x = Dense(2048, activation='relu',name='fc3')(x)
    x = Dense(2048, activation='relu',name='fc4')(x)
    x = Dense(60,activation='linear',name='predictions')(x)
    fcmodel = Model(fc_input,x)
    # Load the model weights if instructed
    if loadweights:
        fcmodel.load_weights(weightfile)
    # Compile and print summary
    fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
        metrics=['accuracy'])
    if not stop_summary:
        print 'Convolutional Part:'
        vggmodel.summary()
        print 'Fully Connected Part:'
        fcmodel.summary()
    print 'Model loaded'
    return vggmodel, fcmodel

# Double dense with batch normalization
def doubledense_bn(loadweights,weightfile,stop_summary):
    # Vgg model without fully connected layer
    vggmodel = vgg16.VGG16(include_top=False)
    # create fully connected layer
    fc_input = Input(shape=(512,5,10))
    x = Flatten(name='flatten')(fc_input)
    
    x = Dense(2048,name='fc1')(x)
    x = BatchNormalization(name='fc1_bn')(x)
    x = Activation('relu',name='fc1_relu')(x)
    
    x = Dense(2048,name='fc2')(x)
    x = BatchNormalization(name='fc2_bn')(x)
    x = Activation('relu',name='fc2_relu')(x)

    x = Dense(2048,name='fc3')(x)
    x = BatchNormalization(name='fc3_bn')(x)
    x = Activation('relu',name='fc3_relu')(x)

    x = Dense(2048,name='fc4')(x)
    x = BatchNormalization(name='fc4_bn')(x)
    x = Activation('relu',name='fc4_relu')(x)

    x = Dense(60,activation='linear',name='predictions')(x)
    fcmodel = Model(fc_input,x)

    # Load the model weights if instructed
    if loadweights:
        fcmodel.load_weights(weightfile)

    # Compile and print summary
    fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
        metrics=['accuracy'])
    if not stop_summary:
        print 'Convolutional Part:'
        vggmodel.summary()
        print 'Fully Connected Part:'
        fcmodel.summary()
    print 'Model loaded'
    return vggmodel, fcmodel

# Double dense with batch normalization
def doubledense_bn_rg(loadweights,weightfile,stop_summary):
    # Vgg model without fully connected layer
    vggmodel = vgg16.VGG16(include_top=False)
    # create fully connected layer
    fc_input = Input(shape=(512,5,10))
    x = Flatten(name='flatten')(fc_input)
    
    x = Dense(2048,W_regularizer=l1(0.01),name='fc1')(x)
    x = BatchNormalization(name='fc1_bn')(x)
    x = Activation('relu',name='fc1_relu')(x)
    
    x = Dense(2048,W_regularizer=l1(0.01),name='fc2')(x)
    x = BatchNormalization(name='fc2_bn')(x)
    x = Activation('relu',name='fc2_relu')(x)

    x = Dense(2048,W_regularizer=l1(0.01),name='fc3')(x)
    x = BatchNormalization(name='fc3_bn')(x)
    x = Activation('relu',name='fc3_relu')(x)

    x = Dense(2048,W_regularizer=l1(0.01),name='fc4')(x)
    x = BatchNormalization(name='fc4_bn')(x)
    x = Activation('relu',name='fc4_relu')(x)

    x = Dense(60,activation='linear',name='predictions')(x)
    fcmodel = Model(fc_input,x)

    # Load the model weights if instructed
    if loadweights:
        fcmodel.load_weights(weightfile)

    # Compile and print summary
    fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
        metrics=['accuracy'])
    if not stop_summary:
        print 'Convolutional Part:'
        vggmodel.summary()
        print 'Fully Connected Part:'
        fcmodel.summary()
    print 'Model loaded'
    return vggmodel, fcmodel


# This is a preset model with four convolutional blocks
def lesscnn (loadweights,weightfile,vggweightfile,\
    stop_summary,nb_cnnblocks):
    # Vgg model without fully connected layer
    vggmodel = custom_vgg16(nb_cnnblocks)
    vggmodel = load_pretrained_weights(vggweightfile,vggmodel)        
    # create fully connected layer
    if nb_cnnblocks == 4:
        fc_input = Input(shape=(512,11,20))
    elif nb_cnnblocks == 3:
        fc_input = Input(shape=(256,22,40))
    elif nb_cnnblocks == 2:
        fc_input = Input(shape=(128,45,80))
    elif nb_cnnblocks == 1:
        fc_input = Input(shape=(64,90,160))
    else:
        raise ValueError('number of cnnblocks must be an int within [1,5)')
    x = Flatten(name='flatten_lesscnn')(fc_input)
    x = Dense(1024, activation='relu',name='fc1_lesscnn')(x)
    x = Dense(1024, activation='relu',name='fc2_lesscnn')(x)
    x = Dense(60,activation='linear',name='predictions_lesscnn')(x)
    fcmodel = Model(fc_input,x)
    # Load the model weights if instructed
    if loadweights:
        fcmodel.load_weights(weightfile)    
    # Compile and print summary
    fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
        metrics=['accuracy'])
    if not stop_summary:
        print 'Convolutional Part:'
        vggmodel.summary()
        print 'Fully Connected Part:'
        fcmodel.summary()
    print 'Model loaded'
    return vggmodel, fcmodel

# Preset model with cnn layers < 5, double dense, batch normalization
# and regularization
def lesscnn_dd_bn_rg (loadweights,weightfile,vggweightfile,\
    stop_summary,nb_cnnblocks):
    # Vgg model without fully connected layer
    vggmodel = custom_vgg16(nb_cnnblocks)
    vggmodel = load_pretrained_weights(vggweightfile,vggmodel)        
    # create fully connected layer
    if nb_cnnblocks == 4:
        fc_input = Input(shape=(512,11,20))
    elif nb_cnnblocks == 3:
        fc_input = Input(shape=(256,22,40))
    elif nb_cnnblocks == 2:
        fc_input = Input(shape=(128,45,80))
    elif nb_cnnblocks == 1:
        fc_input = Input(shape=(64,90,160))
    else:
        raise ValueError('number of cnnblocks must be an int within [1,5)')
    
    # Creating the fully connected layers
    x = Flatten(name='flatten')(fc_input)
    
    #x = Dense(2048,name='fc1')(x)
    #x = BatchNormalization(name='fc1_bn')(x)
    #x = Activation('relu',name='fc1_relu')(x)
    
    x = Dense(1024,activity_regularizer=activity_l1(0.01),name='fc2')(x)
    x = BatchNormalization(name='fc2_bn')(x)
    x = Activation('relu',name='fc2_relu')(x)

    x = Dense(2048,activity_regularizer=activity_l1(0.01),name='fc3')(x)
    x = BatchNormalization(name='fc3_bn')(x)
    x = Activation('relu',name='fc3_relu')(x)

    x = Dense(2048,activity_regularizer=activity_l1(0.01),name='fc4')(x)
    x = BatchNormalization(name='fc4_bn')(x)
    x = Activation('relu',name='fc4_relu')(x)

    x = Dense(60,activation='linear',name='predictions')(x)
    fcmodel = Model(fc_input,x)
    # Load the model weights if instructed
    if loadweights:
        fcmodel.load_weights(weightfile)    
    # Compile and print summary
    fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
        metrics=['accuracy'])
    if not stop_summary:
        print 'Convolutional Part:'
        vggmodel.summary()
        print 'Fully Connected Part:'
        fcmodel.summary()
    print 'Model loaded'
    return vggmodel, fcmodel

# Double dense with batch normalization
def residual_bn_rg(loadweights,weightfile,stop_summary):
    # Vgg model without fully connected layer
    vggmodel = vgg16.VGG16(include_top=False)
    # create fully connected layer
    fc_input = Input(shape=(512,5,10))

    # Reshape layer
    x1 = Flatten(name='flatten')(fc_input)
    x1 = Dense(2048,W_regularizer=l1(0.01),name='fc1')(x1)
    x1 = BatchNormalization(name='fc1_bn')(x1)
    x1 = Activation('relu',name='fc1_relu')(x1)
    
    # Residual Block
    x2 = Dense(2048,W_regularizer=l1(0.01),name='fc2')(x1)
    x2 = BatchNormalization(name='fc2_bn')(x2)
    x2 = Activation('relu',name='fc2_relu')(x2)

    # Sum input with residual
    x3 = merge([x1,x2],mode = 'sum',name='res_1')

    # Residual block
    x4 = Dense(2048,W_regularizer=l1(0.01),name='fc3')(x3)
    x4 = BatchNormalization(name='fc3_bn')(x4)
    x4 = Activation('relu',name='fc3_relu')(x4)

    # Residual block
    x5 = Dense(2048,W_regularizer=l1(0.01),name='fc4')(x4)
    x5 = BatchNormalization(name='fc4_bn')(x5)
    x5 = Activation('relu',name='fc4_relu')(x5)

    # Sum input with residual
    x = merge([x3,x5],mode = 'sum',name='res_2')

    # Reshape layer
    x = Dense(60,activation='linear',name='predictions')(x)
    fcmodel = Model(fc_input,x)

    # Load the model weights if instructed
    if loadweights:
        fcmodel.load_weights(weightfile)

    # Compile and print summary
    fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
        metrics=['accuracy'])
    if not stop_summary:
        print 'Convolutional Part:'
        vggmodel.summary()
        print 'Fully Connected Part:'
        fcmodel.summary()
    print 'Model loaded'
    return vggmodel, fcmodel

# With the four preset VGG16 block, one is kept tunable
def tunable (loadweights,weightfile,vggweightfile,stop_summary):
    # Vgg model without fully connected layer
    vggmodel = custom_vgg16(4)
    vggmodel = load_pretrained_weights(vggweightfile,vggmodel)    
    # Create tunable + fully connected
    fc_input = Input(shape=(512,11,20))
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same',\
     name='block5_conv1')(fc_input)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same',\
     name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same',\
     name='block5_conv3')(x)
    x = Flatten(name='flatten_fourcnn')(x)
    x = Dense(1024, activation='relu',name='fc1_fourcnn')(x)
    x = Dense(1024, activation='relu',name='fc2_fourcnn')(x)
    x = Dense(60,activation='linear',name='predictions_fourcnn')(x)
    fcmodel = Model(fc_input,x)
    # Load the model weights if instructed
    if loadweights:
        fcmodel.load_weights(weightfile)
    # Compile and print summary
    fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
        metrics=['accuracy'])
    if not stop_summary:
        print 'Convolutional Part:'
        vggmodel.summary()
        print 'Fully Connected Part:'
        fcmodel.summary()
    print 'Model loaded'
    return vggmodel,fcmodel

# Returns a custom vgg16 model with specified number of blocks
def custom_vgg16(total_blks):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(3,None,None)))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    if total_blks > 1:
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128,3,3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128,3,3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

    if total_blks > 2:
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256,3,3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256,3,3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256,3,3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

    if total_blks > 3:        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512,3,3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512,3,3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512,3,3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

    if total_blks > 4:
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512,3,3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512,3,3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512,3,3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    return model

# Loads the pretrained weights for a given vgg16 model
def load_pretrained_weights(vggweightfile, model):
    # Open vgg16_weights.h5
    f = h5py.File(vggweightfile)
    # Total number of layers in the model
    n = len(model.layers)
    for k in range(n):
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for \
            p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    return model
