import tensorflow as tf

def load_jpeg():
    # This function should read the JPEG folder and apply image augmentations and return the generator object.
    #Input: None
    #Output: Data and target in a tuple (data,target)
    #just some random comments
    directory= "../input/siim-isic-melanoma-classification/jpeg/"

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=45,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.1,
        dtype=None)
	

    train_ds=data_gen.flow_from_directory(
	    directory,
	    target_size=(256, 256),
	    color_mode="rgb",
	    classes=None,
	    class_mode="categorical",
	    batch_size=32,
	    shuffle=True,
	    seed=None)

    return train_ds
