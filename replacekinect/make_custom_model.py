from keras.layers import Flatten, Dense, Input, Convolution2D, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from learningtools.preset_models import custom_vgg16, load_pretrained_weights


# This code creates a custom VGG and an fc model and saved as json strings
# The CNN model will have 4 CNN blocks from VGG, one tunable CNN block without
# maxpooling layer, 4 layers of dense with 2048 neuron -- each interleaving with
# batch normalization layers. No regularizations

# Filename prefix
prefix = 'cnn4_tunable_dd_bn'

# VGG with first 4 conv blocks
vggmodel = custom_vgg16(4)
vggmodel = load_pretrained_weights('vgg16_weights.h5',vggmodel)

# Create tunable + fully connected
fc_input = Input(shape=(512,11,20))
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same',\
 name='block5_conv1')(fc_input)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same',\
 name='block5_conv2')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same',\
 name='block5_conv3')(x)

x = Flatten(name='flatten_fourcnn')(x)
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

# Compile and print summary
fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
    metrics=['accuracy'])

print 'Convolutional Part:'
vggmodel.summary()
print 'Fully Connected Part:'
fcmodel.summary()


# Save the model
vggmodel.save(prefix+'_cnn.h5')
fcmodel.save(prefix+'_fc.h5')
print 'model saved'
