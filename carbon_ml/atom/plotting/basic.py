import matplotlib.pyplot as plt
import numpy as np


class BasicPlotter:

    def __init__(self):
        pass

    @staticmethod
    def show_learning_curve(train_errors, test_errors, save=False):
        """
        Plots the learning curve of test and train sets.

        :param train_errors: (list) of training error values
        :param test_errors: (list) of testing error values
        :param save: if set to True plot will be saved as file
        :return: None
        """
        plt.figure(figsize=(15, 7))
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(test_errors), "b-", linewidth=3, label="val")
        plt.xlabel("Iterations")
        plt.ylabel('Error')
        plt.title('Learning Curve')
        plt.legend(loc='upper right')
        if save:
            plt.savefig('learning_curve')
        plt.show()
