from keras.applications import vgg16
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from skeletonutils import data_stream_shuffle

# This is a preset model configuration for replacekinect project
# The 

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

    fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',\
        metrics=['accuracy'])
    if not stop_summary:
        print 'Convolutional Part:'
        vggmodel.summary()
        print 'Fully Connected Part:'
        fcmodel.summary()
    print 'Model loaded'

    return vggmodel, fcmodel