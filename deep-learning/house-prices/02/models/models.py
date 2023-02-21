import tensorflow as tf


def l8_mp2_mp12(features_count, mp=4):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 1.5, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 1.2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count, activation=activation))
    model.add(tf.keras.layers.Dense(int(features_count / 2), activation=activation))
    model.add(tf.keras.layers.Dense(1))

    return model


def l8_mp2(features_count, mp=4):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count, activation=activation))
    model.add(tf.keras.layers.Dense(1))

    return model


def l9_mp4(features_count, mp=4):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 4, activation=activation))
    model.add(tf.keras.layers.Dense(features_count, activation=activation))
    model.add(tf.keras.layers.Dense(1))

    return model


def l7_mp2(features_count, mp=4):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count, activation=activation))
    model.add(tf.keras.layers.Dense(1))

    return model


def l6_mp2(features_count, mp=4):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count, activation=activation))
    model.add(tf.keras.layers.Dense(1))

    return model


def l5_mp2(features_count, mp=4):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count * mp * 2, activation=activation))
    model.add(tf.keras.layers.Dense(features_count, activation=activation))
    model.add(tf.keras.layers.Dense(1))

    return model


def l5_mp2_k0(features_count, mp=4):
    # activation = tf.keras.activations.relu
    activation = 'relu'
    kernel_init = tf.keras.initializers.RandomNormal()
    bias_init = tf.keras.initializers.Zeros()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count
                                    , activation=activation
                                    , input_shape=[features_count],
                                    kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))
    model.add(tf.keras.layers.Dense(features_count * mp
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))
    model.add(tf.keras.layers.Dense(features_count * mp * 2
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))
    model.add(tf.keras.layers.Dense(features_count
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))
    model.add(tf.keras.layers.Dense(1
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))

    return model


def l4_mp_reg(features_count, mp=4):
    # activation = tf.keras.activations.relu
    activation = 'relu'
    kernel_init = tf.keras.initializers.RandomNormal()
    bias_init = tf.keras.initializers.Zeros()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count
                                    , activation=activation
                                    , input_shape=[features_count],
                                    kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))
    model.add(tf.keras.layers.Dense(features_count * mp
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))
    model.add(tf.keras.layers.Dense(features_count * mp
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))
    model.add(tf.keras.layers.Dense(1
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))

    return model


def l4_mp4_reg(features_count, mp=4, mp2=2):
    return l4_mp2_reg(features_count, mp, mp2)


def l4_mp2_reg(units_base, input_shape
               , mp=2, mp2=2
               , kernel_init=tf.keras.initializers.GlorotNormal(seed=7)  # Xavier
               , bias_init=tf.keras.initializers.Zeros()
               , activity_regularizer=tf.keras.regularizers.l1(0.001)
               , activation='relu'
               ):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units_base * mp
                                    , activation=activation
                                    , input_shape=[input_shape]
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dense(units_base * mp
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dense(units_base * mp * mp2
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dense(1))

    return model


def l4_mp2_reg_dropout(units_base, input_shape
                       , mp=2, mp2=2
                       , kernel_init=tf.keras.initializers.GlorotNormal(seed=7)  # Xavier
                       , bias_init=tf.keras.initializers.Zeros()
                       , activity_regularizer=tf.keras.regularizers.l1(0.001)
                       , activation='relu'
                       , activation2='relu'
                       , dropout=0.5
                       ):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units_base * mp
                                    , activation=activation
                                    , input_shape=[input_shape]
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units_base * mp
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units_base * mp * mp2
                                    , activation=activation2
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))

    print("Model Summary: ", model.summary())

    return model


def l4_mp2_reg_cust(features_count, mp=2, mp2=2
                    , kernel_init=tf.keras.initializers.RandomNormal()
                    , bias_init=tf.keras.initializers.Zeros()
                    , activity_regularizer=tf.keras.regularizers.l1(0.01)):
    activation = 'relu'

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count
                                    , activation=activation
                                    , input_shape=[features_count]
                                    ))
    model.add(tf.keras.layers.Dense(features_count * mp
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dense(features_count * mp * mp2
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dense(1
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))

    return model


def l4_mp2_reg_drop(features_count, mp=2, mp2=2
                    , kernel_init=tf.keras.initializers.GlorotNormal(seed=7)
                    , bias_init=tf.keras.initializers.Zeros()
                    , activity_regularizer=tf.keras.regularizers.l1(0.001)
                    , dropout=0.5):
    # activation = tf.keras.activations.relu
    activation = 'relu'
    # kernel_init = tf.keras.initializers.RandomNormal()
    # glorot = tf.keras.initializers.GlorotNormal()
    # kernel_init = glorot
    dropout_seed = 7

    # bias_init = tf.keras.initializers.Zeros()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count
                                    , activation=activation
                                    , input_shape=[features_count]
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dropout(dropout, seed=dropout_seed))
    model.add(tf.keras.layers.Dense(features_count * mp
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dropout(dropout, seed=dropout_seed))
    model.add(tf.keras.layers.Dense(features_count * mp * mp2
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))
    model.add(tf.keras.layers.Dropout(dropout, seed=dropout_seed))
    model.add(tf.keras.layers.Dense(1
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init
                                    , activity_regularizer=activity_regularizer))

    return model


def l3_mp2_reg(features_count, mp=4):
    # activation = tf.keras.activations.relu
    activation = 'relu'
    kernel_init = tf.keras.initializers.RandomNormal()
    bias_init = tf.keras.initializers.Zeros()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count
                                    , activation=activation
                                    , input_shape=[features_count],
                                    kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))
    model.add(tf.keras.layers.Dense(features_count * mp * 2
                                    , activation=activation
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))
    model.add(tf.keras.layers.Dense(1
                                    , kernel_initializer=kernel_init
                                    , bias_initializer=bias_init))

    return model


def l4_mp(features_count, mp=4):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(features_count, activation=activation))
    model.add(tf.keras.layers.Dense(1))

    return model


def l3_mp(features_count, mp=4):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation))
    model.add(tf.keras.layers.Dense(1))

    return model


def l3(features_count):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(features_count, activation=activation))
    model.add(tf.keras.layers.Dense(1))

    return model


def l2_mp(features_count, mp=2):
    activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(features_count * mp, activation=activation, input_shape=[features_count]))
    model.add(tf.keras.layers.Dense(1))

    return model
