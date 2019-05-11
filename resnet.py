import numpy as np 
import tensorflow as tf 
from callbacks import resnet_lr_schedule
import os

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = tf.keras.layers.Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
        x = conv(x)
    return x

def resnet_model(input_shape=None, depth=None, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = tf.keras.layers.Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    y = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def resnet(x_train=None,x_test=None,y_train=None,y_test=None,num_classes=10,batch_size=32,epochs=5,subtract_pixel_mean=True,n=3):
    depth = n*9 + 2

    input_shape = x_train.shape[1:]

    #Normalise the data
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

    #If subtracting the pixel mean
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train,axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    #Convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train,num_classes=num_classes)
    y_test  = tf.keras.utils.to_categorical(y_test,num_classes=num_classes)

    model = resnet_model(input_shape=input_shape,depth=depth,num_classes=num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=resnet_lr_schedule(0)),
                  metrics=['accuracy'])
    
    # model.summary()

    #Prepare save structure
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'resnet.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir,model_name)

    #Prepare callbacks for model saving, and for learning rate adjustment
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                    monitor='acc',
                                                    save_best_only=True)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(resnet_lr_schedule)

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                      cooldown=0,
                                                      monitor='loss',
                                                      patience=5,
                                                      min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    #Run training
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=callbacks)

    #Generate predictions
    predictions = model.predict(x_test)
    if len(y_test.shape)==2:
        correct = [True if np.argmax(pred)==true[0] else False for pred,true in zip(predictions,y_test)]
    elif len(y_test.shape)==1:
        correct = [True if np.argmax(pred)==true else False for pred,true in zip(predictions,y_test)]
    print('Number incorrect: {}'.format(len(correct)-np.sum(correct)))
    print('Accuracy on test: {:.3f}'.format(np.mean(correct)))

