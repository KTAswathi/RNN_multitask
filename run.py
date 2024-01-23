import torch
from RNN_rate_dynamics import RNNLayer

T, batch = 1000, 100
n_input, n_rnn, n_output = 10, 500, 5

rnn  = RNNLayer(n_input, n_rnn, torch.nn.ReLU(), 0.9, True)   # input_size, hidden_size, nonlinearity, decay, bias

import multitask

hp, log, optimizer = multitask.set_hyperparameters(model_dir='debug', hp={'learning_rate': 0.001}, ruleset='all') #, rich_output=True)
run_model = multitask.Run_Model(hp, RNNLayer)
multitask.train(run_model, optimizer, hp, log)

import matplotlib.pyplot as plt

# Extract data for 'contextdm1' and 'contextdm2'
cost_contextdm1 = log['cost_contextdm1']
perf_contextdm1 = log['perf_contextdm1']
cost_contextdm2 = log['cost_contextdm2']
perf_contextdm2 = log['perf_contextdm2']

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plotting Cost for 'contextdm1'
axes[0, 0].plot(cost_contextdm1, label='Cost for contextdm1')
axes[0, 0].set_title('Cost Over Time for contextdm1')
axes[0, 0].set_ylabel('Cost')
axes[0, 0].legend()

# Plotting Performance for 'contextdm1'
axes[0, 1].plot(perf_contextdm1, label='Performance for contextdm1')
axes[0, 1].set_title('Performance Over Time for contextdm1')
axes[0, 1].set_ylabel('Performance')
axes[0, 1].legend()

# Plotting Cost for 'contextdm2'
axes[1, 0].plot(cost_contextdm2, label='Cost for contextdm2')
axes[1, 0].set_title('Cost Over Time for contextdm2')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Cost')
axes[1, 0].legend()

# Plotting Performance for 'contextdm2'
axes[1, 1].plot(perf_contextdm2, label='Performance for contextdm2')
axes[1, 1].set_title('Performance Over Time for contextdm2')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].set_ylabel('Performance')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

