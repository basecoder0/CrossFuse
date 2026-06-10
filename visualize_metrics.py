import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":


    data_ir = scipy.io.loadmat('models/autoencoder/loss/loss_data_trans_ir_e3.mat')
    data_vi = scipy.io.loadmat('models/autoencoder/loss/loss_data_trans_vi_e3.mat')
    data_fuse = scipy.io.loadmat('models/transfuse/loss/loss_data_trans_e31.mat')
    # loss_mat_ir = data_ir['loss_data'].flatten()
    # loss_mat_vi = data_vi['loss_data'].flatten()
    loss_mat_fuse = data_fuse['loss_data'].flatten()
    plt.figure()
    # plt.plot(loss_mat_ir, label='IR Loss')
    # plt.plot(loss_mat_vi, label='VI Loss')
    plt.plot(loss_mat_fuse, label='Fused Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig('models/transfuse/loss/loss_curve_epoch_3.png')
    plt.close()

    plt.show()
