import numpy as np

from d3qn import Agent
from Utils.atari_utils import make_env
from Utils.plot_learning_curve import plot_learning_curve


def play_one_episode(agent, env, training):
    observation = env.reset()
    done = False
    total_reward = 0.0
    counter = 0
    lives = 3

    while not done:
        a = agent.get_action(state=observation, training=training)
        prev_observation = observation
        observation, reward, done, info = env.step(a)
        total_reward += reward
        counter += 1

        if training:
            if info['lives'] < lives:
                lives = info['lives']
                agent.replay_buffer.add_to_mem(state=prev_observation, action=a, reward=reward,
                                               state_new=observation, done=True)
            else:
                lives = info['lives']
                agent.replay_buffer.add_to_mem(state=prev_observation, action=a, reward=reward,
                                               state_new=observation, done=done)
            if agent.replay_buffer.counter > agent.batch_size:
                agent.update_networks()

    return agent, total_reward


def main(training):
    env_name = 'MsPacmanNoFrameskip-v4'
    env = make_env(env_name, no_ops=30)
    chkpt_dir = env_name

    memory_size = 75000
    learning_rate = 0.0001
    batch_size = 32
    gamma = 0.99
    initial_eps = 1.0
    final_eps = 0.1
    final_eps_state = 500000
    update_frequency = 1
    target_update_frequency = 1000
    scale_gradients = False
    agent = Agent(state_dims=env.observation_space.shape, n_actions=env.action_space.n, frame_as_input=True,
                  memory_size=memory_size, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                  initial_eps=initial_eps, final_eps=final_eps, final_eps_state=final_eps_state,
                  update_frequency=update_frequency, target_update_frequency=target_update_frequency,
                  scale_gradients=scale_gradients)

    if training:
        scores_train = []
        avg_scores_train = []
        scores_test = []
        n_episodes = 3000
        test_frequency = 50
        best_test_score = -np.inf

        for e in range(n_episodes):
            agent, total_reward_train = play_one_episode(agent, env, training=True)
            scores_train.append(total_reward_train)
            avg_scores_train.append(np.mean(scores_train[-100:]))
            print('episode #{} -----> training score:{:.2f} | average score:{:.2f} | best test score:{:.2f}'
                  .format(e + 1, total_reward_train, avg_scores_train[-1], best_test_score))

            if total_reward_train > best_test_score:
                agent, total_reward_test = play_one_episode(agent, env, training=False)
                scores_test.append([e + 1, total_reward_test])
                if total_reward_test >= best_test_score:
                    best_test_score = total_reward_test
                    agent.save_networks(chkpt_dir)

            if (e + 1) % test_frequency == 0 or e == 0:
                agent, total_reward_test = play_one_episode(agent, env, training=False)
                scores_test.append([e + 1, total_reward_test])
                print('-------------------------------> test score:{:.2f} <-------------------------------\n'
                      .format(total_reward_test))

                if total_reward_test >= best_test_score:
                    best_test_score = total_reward_test
                    agent.save_networks(chkpt_dir)

        # Plot the learning curve
        plot_learning_curve(env_name=env_name, directory=chkpt_dir, training_scores=scores_train,
                            avg_training_scores=avg_scores_train, test_scores=scores_test)

    else:
        env = make_env(env_name, render_mode='human')

        # Load the trained networks
        agent.load_networks(chkpt_dir)

        # Show the video
        for e in range(1):
            agent, total_reward_test = play_one_episode(agent, env, training=False)
            print('video #{} -----> total reward:{:.2f}'.format(e + 1, total_reward_test))
        env.close()


if __name__ == '__main__':
    main(training=False)
