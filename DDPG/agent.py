import numpy as np
import tensorflow as tf
from models import critic_gen, actor_gen
from buffer import BasicBuffer_a
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import random


class DDPGAgent:

    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        print(env.action_space.high[0])

        self.gamma = gamma
        self.tau = tau

        actor_layer = [512, 512, 300, 200, 128]
        critic_layer = [1024, 512, 300, 1]
        self.actor_layer = actor_layer
        self.critic_layer = critic_layer

        self.mu = actor_gen(self.obs_dim, self.action_dim, actor_layer, self.action_max)
        self.q_mu = critic_gen(self.obs_dim, self.action_dim, critic_layer)

        self.mu_target = actor_gen(self.obs_dim, self.action_dim, actor_layer, self.action_max)
        self.q_mu_target = critic_gen(self.obs_dim, self.action_dim, critic_layer)

        self.mu_target.set_weights(self.mu.get_weights())
        self.q_mu_target.set_weights(self.q_mu.get_weights())

        self.mu_optimizer = RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01)
        self.q_mu_optimizer = RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01)

        self.replay_buffer = BasicBuffer_a(size=buffer_maxlen, obs_dim=self.obs_dim, act_dim=self.action_dim)

        self.q_losses = []

        self.mu_losses = []

    def get_action(self, s, noise_scale=0, eps=1):
        s = np.asarray(s)
        a = self.mu.predict(s.reshape(1, -1))[0]
        a += random.randint(-1, 1) * noise_scale * np.random.randn(self.action_dim)
        return np.clip(a, 0, self.action_max)


    def update(self, batch_size):
        X, A, R, X2, D = self.replay_buffer.sample(batch_size)
        X = np.asarray(X, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)
        X2 = np.asarray(X2, dtype=np.float32)

        Xten = tf.convert_to_tensor(X)

        # Updating Critic
        with tf.GradientTape() as tape:
            A2 = self.mu_target(X2)
            q_target = R + self.gamma * self.q_mu_target([X2, A2])
            qvals = self.q_mu([X, A])
            q_loss = tf.reduce_mean((qvals - q_target) ** 2)
            grads_q = tape.gradient(q_loss, self.q_mu.trainable_variables)
        self.q_mu_optimizer.apply_gradients(zip(grads_q, self.q_mu.trainable_variables))
        self.q_losses.append(q_loss)

        # Updating Actor
        with tf.GradientTape() as tape2:
            A_mu = self.mu(X)
            Q_mu = self.q_mu([X, A_mu])
            mu_loss = -tf.reduce_mean(Q_mu)
            grads_mu = tape2.gradient(mu_loss, self.mu.trainable_variables)
        self.mu_losses.append(mu_loss)
        self.mu_optimizer.apply_gradients(zip(grads_mu, self.mu.trainable_variables))

        temp1 = np.array(self.q_mu_target.get_weights(), dtype=object)
        temp2 = np.array(self.q_mu.get_weights(), dtype=object)
        temp3 = self.tau * temp2 + (1 - self.tau) * temp1
        self.q_mu_target.set_weights(temp3)

        temp1 = np.array(self.mu_target.get_weights(), dtype=object)
        temp2 = np.array(self.mu.get_weights(), dtype=object)
        temp3 = self.tau * temp2 + (1 - self.tau) * temp1
        self.mu_target.set_weights(temp3)

    def save(self, name):
        self.mu_target.save("./mu_models/"+name)
        self.q_mu_target.save("./q_mu_models/"+name)

    def load(self, name):
        self.mu = load_model("./mu_models/"+name, compile=False)
        self.q_mu = load_model("./q_mu_models/"+name, compile=False)