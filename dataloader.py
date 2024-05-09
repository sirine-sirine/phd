
import numpy as np
import torch



class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def transform(self, data: np.array):
        return (data - self.mean) / self.std





dataset_path="C:/Users/user/work_phd/TransProject/abide.npy"

def load_abide_data():

    data = np.load(dataset_path, allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']

    scaler = StandardScaler(mean=np.mean(final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]
    
    
    return final_timeseires, final_pearson, labels, site 

# x,y,z,w=load_abide_data()
# print(x.shape)


