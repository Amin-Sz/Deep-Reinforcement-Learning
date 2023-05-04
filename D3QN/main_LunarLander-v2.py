import numpy as np
import gym

from d3qn import Agent
from Utils.plot_learning_curve import plot_learning_curve


def play_one_episode(agent, env, training, render=False):
    observation = env.reset()
    done = False
    total_reward = 0.0
    counter = 0

    while not done:
        if render:
            env.render()

        a = agent.get_action(state=observation, training=training)
        prev_observation = observation
        observation, reward, done, info = env.step(a)
        total_reward += reward
        counter += 1

        if training:
            if counter >= env._max_episode_steps:
                agent.replay_buffer.add_to_mem(state=prev_observation, action=a, reward=reward,
                                               state_new=observation, done=False)
            else:
                agent.replay_buffer.add_to_mem(state=prev_observation, action=a, reward=reward,
                                               state_new=observation, done=done)

            if agent.replay_buffer.counter > agent.batch_size:
                agent.update_networks()

    return agent, total_reward


def main(training):
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    chkpt_dir = env_name

    memory_size = 1000000
    learning_rate = 0.001
    batch_size = 32
    gamma = 0.99
    initial_eps = 1.0
    final_eps = 0.01
    final_eps_state = 200000
    update_frequency = 1
    target_update_frequency = 1000
    scale_gradients = False
    agent = Agent(state_dims=env.observation_space.shape, n_actions=env.action_space.n, frame_as_input=False,
                  memory_size=memory_size, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                  initial_eps=initial_eps, final_eps=final_eps, final_eps_state=final_eps_state,
                  update_frequency=update_frequency, target_update_frequency=target_update_frequency,
                  scale_gradients=scale_gradients)

    if training:
        scores_train = []
        avg_scores_train = []
        scores_test = []
        n_episodes = 1500

        for e in range(n_episodes):
            agent, total_reward_train = play_one_episode(agent, env, training=True)
            scores_train.append(total_reward_train)
            avg_scores_train.append(np.mean(scores_train[-100:]))
            print('episode #{} -----> training score:{:.2f} | average score:{:.2f}'
                  .format(e + 1, total_reward_train, avg_scores_train[-1]))

        agent.save_networks(chkpt_dir)

        # Plot the learning curve
        plot_learning_curve(env_name=env_name, directory=chkpt_dir, training_scores=scores_train,
                            avg_training_scores=avg_scores_train, test_scores=scores_test)

    else:
        env = gym.make(env_name)

        # Load the trained networks
        agent.load_networks(chkpt_dir)

        # Show the video
        for e in range(20):
            agent, total_reward_test = play_one_episode(agent, env, training=False, render=True)
            print('video #{} -----> total reward:{:.2f}'.format(e + 1, total_reward_test))
        env.close()


if __name__ == '__main__':
    main(training=False)
