import matplotlib.pyplot as plt
from scipy.stats import probplot, norm


class VisualTests():
    def __init__(self, var_names):
        """
        Class for organizing multiple plots
        """
        self.n_rows = len(var_names)
        self.var_names = var_names  # Store variable names for plots?
        self.fig, self.axes = plt.subplots(self.n_rows, 2, figsize=(10, 4 * self.n_rows))
        if self.n_rows == 1:
            # Fix to handle also only one dim.
            self.axes = self.axes.reshape(1, -1)
        self.last_plot_row_idx = None

    def next_row(self):
        """
        Method for fetching the right axes[i, j] to plot into.
        """
        if self.last_plot_row_idx is None:
            self.last_plot_row_idx = 0
            return self.last_plot_row_idx
        else:
            self.last_plot_row_idx += 1
            return self.last_plot_row_idx

    def plot(self, observations_A, observations_B,
             var_name, task='Posterior Predictive',
             group_name_A='Low-dim',
             group_name_B='Full plate'):
            """
            Assuming that observations in A are for the low-dimensional version.

            Args:
            observations_A (np.array): Observations from generating process A,
            Assumed to be 
            observations_B (np.array): Observations from generating process B
            task (str): Name of task to use as label in plots and prints. 
            Typically 'Prior Predictive', 'Posterior', or 'Posterior Predictive'
            var_name (str): Name of the variable being checked (for plots and 
            prints).
            group_name_A (str): Descriptive name of generating process A. 
            Defaults to 'Low-dim'.
            group_name_B (str): Descriptive name of generating process A. 
            Defaults to 'Full plate'.
            """
            row_idx = self.next_row()
            # Overlapping histograms:
            self.axes[row_idx, 0].hist(observations_A, bins=50, alpha=0.5, label=group_name_A, density=True)
            self.axes[row_idx, 0].hist(observations_B, bins=50, alpha=0.5, label=group_name_B, density=True)
            self.axes[row_idx, 0].legend()
            self.axes[row_idx, 0].set_xlabel(var_name)
            self.axes[row_idx, 0].set_title('Prior Predictive of {}'.format(var_name))
            # Q-Q plot
            # PLOT AGAINST WHAT? SHOULD THERE NOT BE OBSERVATIONS_B IN THERE?
            probplot(observations_A, dist=norm, plot=self.axes[row_idx, 1])  # Why is this against "theoretical quantiles"?
            self.axes[row_idx, 1].set_title('Q-Q plot of {}'.format(group_name_A))

    def show(self):
        plt.tight_layout()
        plt.show()
