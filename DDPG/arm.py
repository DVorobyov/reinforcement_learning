from osim.env.arm import Arm2DVecEnv
import math
from agent import DDPGAgent




def count_dist(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def trainer(env, agent, max_episodes, max_steps, min_eps, batch_size, eps, decr_rate, action_noise):
    episode_rewards = []
    best_model = 0
    best_score = 0
    best_episode = 0

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        if action_noise > min_eps:
            action_noise *= decr_rate
        for t in range(max_steps):
            step += 1
            action = agent.get_action(state, action_noise)

            next_state, reward, done, _ = env.step(action)
            d_store = False if step == max_steps-1 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)
            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward) + " eps: " + str(action_noise) + " steps: " + str(step))
                if episode_reward >= best_score:
                    best_model = agent
                    best_episode = episode
                    if episode_reward >= 195:
                        print(f"Saving trained model as arm{episode}-{episode_reward}.h5")
                        agent.save(f"arm{episode}-{episode_reward}-{batch_size}.h5")
                break

            state = next_state
        if agent.replay_buffer.size > batch_size:
            for i in range(int(step/4)):
                agent.update(batch_size)

    best_model.save(f"arm{best_episode}-{best_score}-{batch_size}.h5")
    s_for_name = ""
    for l in agent.actor_layer:
        s_for_name += "-" + str(l)
    s_for_name += "_"
    for l in agent.critic_layer:
        s_for_name += "-" + str(l)
    s_for_name += "_" + str(max_episodes)
    s_for_name += "_" + str(batch_size)
    with open("rewards_arm" + s_for_name + ".txt", "w") as file:
        for r in episode_rewards:
            print(r, file=file)

    return episode_rewards


def test(env, agent, max_episodes, max_steps, batch_size, action_noise):
    rews = []
    agent.load("arm.h5")
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


env = Arm2DVecEnv(visualize=True)
max_episodes = 500
max_steps = 200
batch_size = 1000

gamma = 0.99
tau = 1e-2
buffer_maxlen = 500000
critic_lr = 1e-3
actor_lr = 1e-3

eps = 1
decr_rate = 0.99
min_eps = 0.3

agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
episode_rewards = trainer(env, agent, max_episodes, max_steps, min_eps, batch_size, eps, decr_rate, action_noise=eps)
