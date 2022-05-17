import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

sns.set_theme()
sns.set_context("paper")
sns.axes_style()
sns.despine()
sns.set(font_scale=1.2)

data = pd.read_csv('EMNIST 1 7.csv')

plot = sns.pairplot(data,
            x_vars=['Clean Acc', 'Acc backdoor', 'Backdoor source', 'Backdoor target'],
            y_vars=['LR', 'epsilon','epochs'],
            height=1.8,
            # aspect=1.2
            plot_kws={'s': 60}
            )

plot.axes[2, 0].set_yticks([1, 10, 25, 50, 75, 90, 100])
plot.axes[1, 0].set_yticks([0.1, 0.01, 0.001])
plot.axes[0, 0].set_yticks([0.01, 0.001, 0.0001, 0.00001])

plot.axes[0, 0].set_yscale('log')
plot.axes[1, 0].set_yscale('log')

plot.axes[0, 0].set_ylabel('Learning rate')
plot.axes[1, 0].set_ylabel(r'$\epsilon$')
plot.axes[2, 0].set_ylabel('Number of epochs')

plot.axes[2, 0].set_xlabel('Clean \n accuracy')
plot.axes[2, 1].set_xlabel('Backdoor \n accuracy')
plot.axes[2, 2].set_xlabel('Source class \n accuracy')
plot.axes[2, 3].set_xlabel('Target class \n backdoor accuracy')

plt.savefig('pairplot_EMNIST_1_7.pdf', bbox_inches='tight')
