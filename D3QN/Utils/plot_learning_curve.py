import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(env_name, directory, training_scores, avg_training_scores,
                        test_scores=None, test_window_size=10):
    if not test_scores:
        # axes = plt.axes()
        # axes.set_ylim([np.min(training_scores) - 5, np.max(training_scores) + 5])
        plt.figure(figsize=[8.4, 5.8])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.plot(np.arange(1, len(training_scores) + 1), training_scores, '-', color='xkcd:mid blue', alpha=0.3)
        plt.plot(np.arange(1, len(training_scores) + 1), avg_training_scores, '-', color='xkcd:mid blue')
        legend_2 = 'Running average of the last 100 training scores (' + '%.2f' % np.mean(training_scores[-100:]) + ')'
        plt.legend(['Training score', legend_2], loc=4)
        plt.savefig(directory + '/Rewards_' + env_name)
        plt.show()

    else:
        test_scores = np.array(test_scores)
        avg_test_scores = []
        for t in range(len(test_scores)):
            avg_test_scores.append(np.mean(test_scores[max(0, t - (test_window_size - 1)):t + 1, 1]))

        plt.figure(figsize=[8.4, 5.8])
        # axes = plt.axes()
        # axes.set_ylim([np.min(training_scores) - 5, np.max(test_scores[:, 1]) + 5])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.plot(np.arange(1, len(training_scores) + 1), training_scores, '-', color='xkcd:mid blue', alpha=0.3)
        plt.plot(np.arange(1, len(training_scores) + 1), avg_training_scores, '-', color='xkcd:mid blue')

        plt.plot(test_scores[:, 0], test_scores[:, 1], 'o--', color='xkcd:dull orange', alpha=0.3)
        plt.plot(test_scores[:, 0], avg_test_scores, '-', color='xkcd:dull orange', linewidth=2)

        legend_2 = 'Running average of the last 100 training scores (' + '%.2f' % np.mean(training_scores[-100:]) + ')'
        legend_4 = 'Running average of the last ' + str(test_window_size) + ' test scores (' + \
                   '%.2f' % np.array(avg_test_scores[-1]) + ')'
        plt.legend(['Training score', legend_2, 'Test score', legend_4], loc=4)
        plt.savefig(directory + '/Rewards_' + env_name)
        plt.show()
