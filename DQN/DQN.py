import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop


def OurModel(input_shape, action_space):
    X_input = Input(input_shape)

    X = Dense(64, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='LunarLanderDQNmodel')
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model


class DQNAgent:
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=500000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.996
        self.batch_size = 64
        self.train_start = 1000

        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        total_rewards = []
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            total_reward = 0
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                print(state)
                next_state = np.reshape(next_state, [1, self.state_size])
                total_reward += reward
                if not done:
                    reward = reward
                else:
                    reward = -20
                total_reward += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"episode: {e}/{self.EPISODES}, score: {total_reward}, e: {self.epsilon}")
                    total_rewards.append(total_reward)
                    if total_reward >= 90:
                        print(f"Saving trained model as lunarlander{e}.h5")
                        self.save(f"models_lunar/lunarlander{e}-{total_reward}.h5")
                self.replay()
            if len(self.memory) > self.train_start:
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
        self.save("lunarlander.h5")
        with open('lunarrewards.txt', 'w') as file:
            for r in total_rewards:
                print(r, file=file)

    def test(self):
        rews = []
        self.load("models_lunar/lunarlander365-130.23391589605552.h5")
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            total_r = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                total_r += reward
                state = np.reshape(next_state, [1, self.state_size])
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, total_r))
                    rews.append(total_r)
                    break
        print(sum(rews)/len(rews))


if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()