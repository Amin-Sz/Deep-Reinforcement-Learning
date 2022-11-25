import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(env_name, directory, scores, avg_scores):
    plt.figure(figsize=[8.4, 5.8])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.plot(np.arange(1, len(scores) + 1), scores, '-', color='xkcd:mid blue', alpha=0.3)
    plt.plot(np.arange(1, len(scores) + 1), avg_scores, '-', color='xkcd:mid blue')

    legend_2 = 'Running average of the last 100 scores (' + '%.2f' % np.mean(scores[-100:]) + ')'
    plt.legend(['Training score', legend_2], loc=4)

    plt.savefig(directory + '/Rewards_' + env_name)
    plt.show()
