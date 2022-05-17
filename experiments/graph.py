from http import client, server
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(figsize=(10, 10))

sns.set_theme()
sns.set_context("paper")
sns.axes_style()
sns.despine()

server = np.load('avg_server_acc.npy')
clients = np.load('avg_acc_test.npy')

# vg_acc_test shape is (10, 5, 50)
# Plot the average accuracy of each client taking into account the standard deviation
 
for i in range(5):
    mean = np.mean(clients[:, i, :].squeeze(), axis=0)
    std = np.std(clients[:, i, :].squeeze(), axis=0)

    lower = np.add(mean, -1 * std)
    upper = np.add(mean, std)

    ax.plot(mean, '--', label='Client {}'.format(i + 1))
    ax.fill_between(range(len(mean)), lower, upper, alpha=0.2)

# server = avg_acc_test[:, 0, :]
server = server.squeeze()

mean = np.mean(server, axis=0)
std = np.std(server, axis=0)

lower = np.add(mean, -1 * std)
upper = np.add(mean, std)

ax.plot(mean, '-', label='Server')
ax.fill_between(range(len(mean)), upper, lower, alpha=0.2)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='28')
plt.savefig('mean_std_FMNIST.pdf')