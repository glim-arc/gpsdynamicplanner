import matplotlib.pyplot as plt
import numpy as np
import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())


def plot(model_path):
    avg_loss_list = np.load(os.path.join(model_path, 'obs_avg_loss_list.npy'))
    val_loss = np.load(os.path.join(model_path, 'obs_val_loss.npy'))

    plt.figure()
    epoch = np.arange(1, len(avg_loss_list) + 1)
    plt.plot(epoch, avg_loss_list)
    # plt.legend(["30 Ep", "60 Ep", "100 Ep"])
    plt.ylabel('Average Loss')
    plt.xlabel('Epoch')
    plt.title('CAE Obs Average Loss with validation average loss of  {:.7f}'.format(val_loss / 100))
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'obs_avg_loss_list.jpg'), dpi=200)
    plt.show()

def gps_plot(model_path):
    avg_loss_list = np.load(os.path.join(model_path, 'gps_avg_loss_list.npy'))
    val_loss = np.load(os.path.join(model_path, 'gps_val_loss.npy'))

    plt.figure()
    epoch = np.arange(1, len(avg_loss_list) + 1)
    plt.plot(epoch, avg_loss_list)
    # plt.legend(["30 Ep", "60 Ep", "100 Ep"])
    plt.ylabel('Average Loss')
    plt.xlabel('Epoch')
    plt.title('AE GPS Average Loss with validation average loss of  {:.7f}'.format(val_loss / 100))
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'gps_avg_loss_list.jpg'), dpi=200)
    plt.show()

def planner_plot(model_path, val_loss):
    avg_loss_list = np.load(os.path.join(model_path, 'planner_avg_loss_list.npy'))

    plt.figure()
    epoch = np.arange(1, len(avg_loss_list) + 1)
    plt.plot(epoch, avg_loss_list)
    plt.ylabel('Average Loss')
    plt.xlabel('Epoch')
    plt.title('Planner Average Loss with validation average loss of  {:.7f}'.format(val_loss))
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'planner_loss.jpg'), dpi=200)
    plt.show()

if __name__ == '__main__':
    gps_plot("./mpnet_data/models") 
    plot("./mpnet_data/models")
    val_loss = np.load(os.path.join("./mpnet_data/models", 'planner_val_loss.npy'))
    planner_plot("./mpnet_data/models", val_loss)