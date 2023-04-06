import matplotlib.pyplot as plt
import numpy as np
import os

def plot():
    model_path = "./models"
    avg_loss_list = np.load(os.path.join(model_path,'obs_avg_loss_list.npy'))
    val_loss = np.load(os.path.join(model_path,'obs_val_loss.npy'))

    plt.figure()
    epoch = np.arange(1, len(avg_loss_list) + 1)
    plt.plot(epoch, avg_loss_list)
	# plt.legend(["30 Ep", "60 Ep", "100 Ep"])
    plt.ylabel('Average Loss')
    plt.xlabel('Epoch')
    plt.title('CAE Average Loss with validation average loss of  {:.7f}'.format(val_loss/100))
    plt.savefig(os.path.join(model_path,'obs_avg_loss_list.jpg'), dpi=200)
    plt.show()

plot()