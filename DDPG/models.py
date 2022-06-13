import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import RMSprop
import os

os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = 'C:\\Users\\Dmitry\\PycharmProjects\\DDPG\\rep'
tf.config.experimental.enable_mlir_graph_optimization()


def critic_gen(state_size, action_size, hidden_layers):
    input_x = Input(shape=state_size)
    input_a = Input(shape=action_size)
    x = input_x
    for i, j in enumerate(hidden_layers[:-1]):
        if i == 1:
            x = concatenate([x, input_a], axis=-1)
        x = Dense(j, activation='relu')(x)
    x = Dense(hidden_layers[-1])(x)
    model = tf.keras.Model([input_x, input_a], x)
    optim = RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01)
    model.compile(loss="mse", optimizer=optim, metrics=["accuracy"])

    return model


def actor_gen(state_size, action_size, hidden_layers, action_mult=1):
    input_x = Input(shape=state_size)
    x = input_x
    for i in hidden_layers:
        x = Dense(i, activation='relu')(x)
    x = Dense(action_size, activation='tanh')(x)
    x = tf.math.multiply(x, action_mult)

    model = tf.keras.Model(inputs=input_x, outputs=x, name='LunarLanderDQNmodel_actor')
    optim = RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01)
    model.compile(loss="mse", optimizer=optim, metrics=["accuracy"])

    model.summary()
    return model
