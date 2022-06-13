import gym
import math
from agent import DDPGAgent



def count_dist(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def trainer(env, agent, max_episodes, max_steps, min_eps, batch_size, eps, decr_rate, action_noise):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        if action_noise > min_eps:
            action_noise *= decr_rate
        while not done:
            step += 1
            agent.env.render()
            action = agent.get_action(state, action_noise)

            next_state, reward, done, _ = env.step(action)

            d_store = False if step == max_steps-1 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)
            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward) + " eps: " + str(action_noise))
                if episode_reward >= 100:
                    print(f"Saving trained model as cartpole{episode}-{episode_reward}.h5")
                    agent.save(f"cartpole{episode}-{episode_reward}-{batch_size}.h5")
                break

            state = next_state

        if agent.replay_buffer.size > batch_size:
            for i in range(int(agent.replay_buffer.size / 500)):
                agent.update(batch_size)


    s_for_name = ""
    for l in agent.actor_layer:
        s_for_name += "-" + str(l)
    s_for_name += "_"
    for l in agent.critic_layer:
        s_for_name += "-" + str(l)
    s_for_name += "_" + str(max_episodes)
    s_for_name += "_" + str(batch_size)
    with open("lunarlander" + s_for_name + ".txt", "w") as file:
        for r in episode_rewards:
            print(r)

    return episode_rewards


def test(env, agent, max_episodes, max_steps, batch_size, action_noise):
    rews = []
    agent.load("lunarlander988-253.1572060422451.h5")
    for e in range(100):
        state = agent.env.reset()
        done = False
        total_r = 0
        while not done:
            agent.env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            total_r += reward

            if done:
                print("episode: {}/{}, score: {}".format(e, 100, total_r))
                rews.append(total_r)
                break
        state = next_state
    print(sum(rews) / len(rews))


env = gym.make("MountainCarContinuous-v0")
max_episodes = 500
max_steps = 500
batch_size = 256

gamma = 0.99
tau = 1e-2
buffer_maxlen = 500000
critic_lr = 1e-3
actor_lr = 1e-3

eps = 1
decr_rate = 0.995
min_eps = 0.3

agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
episode_rewards = trainer(env, agent, max_episodes, max_steps, min_eps, batch_size, eps, decr_rate, action_noise=1)