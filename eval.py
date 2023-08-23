# load the model
import os
import pickle
from params import *
import matplotlib.pyplot as plt

# load the training history
training_history = pickle.load(open(os.path.join(results_dir, 'training_histories.pickle'), 'rb'))

plt.plot(training_history['loss_train'])
plt.plot(training_history['loss_val'])
plt.show()
