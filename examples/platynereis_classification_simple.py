import os.path as osp
import os
import argparse
import glob
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from torch_geometric.nn import DeepGCNLayer, GENConv
import pandas as pd
import numpy as np
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='./', type='train', size=None, transform=None):
        self.nuclei_detections_list = sorted(glob.glob(osp.join(data_dir, type, '*.csv')))  # TODO
        self.type=type
        self.size = size
        self.real_size = len(self.nuclei_detections_list)
        print("Number of files in {} dataset  is {}".format(type, self.real_size))
        self.transform = transform

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def _normalize(self, data):
        data = torch.from_numpy(data).float()
        data_mean = data.mean(dim=0, keepdim=True)
        data = data - data_mean
        data_norm = data.norm(dim=1).max()
        data = data / data_norm
        return data

    
    def __obtain_transformed_torch_nuclei(self, index, nuclei_csv):
        
        nuclei_detections = pd.read_csv(nuclei_csv[index], delimiter=',', header=None, usecols=[1, 2, 3])  # x, y, z are first, second and third columns
        nuclei_detections = nuclei_detections.to_numpy()
        nuclei_ids = pd.read_csv(nuclei_csv[index], delimiter=',', header=None, usecols=[4])  # class id is 4th column
        nuclei_ids = nuclei_ids.to_numpy()
        if self.type=='test':
            nuclei_detections_torch = self._normalize(nuclei_detections)  # normalized
        else:
            #nuclei_detections_rotated, nuclei_indices = self._transform(nuclei_detections)
            #nuclei_ids = nuclei_ids[nuclei_indices] 
            #nuclei_detections_torch = self._normalize(nuclei_detections_rotated)  # normalized
            nuclei_detections_torch = self._normalize(nuclei_detections)  # normalized
        
        return nuclei_detections_torch, torch.from_numpy(nuclei_ids).long()

    def __getitem__(self, idx):
        index = idx if self.size is None else random.randint(0, self.real_size - 1)
        nuclei_detections_torch, nuclei_ids = self.__obtain_transformed_torch_nuclei(index, self.nuclei_detections_list)
        
        # now add some noise (which is a function of pairwise distance)
        #std = 0.02  # TODO
        #noise = np.random.normal(0, std, size=nuclei_detections_torch.shape)
        #noise = np.clip(noise, a_min=-5 * std, a_max=5 * std)
        #if self.type=='test':
        #    pass
        #else:
        #    nuclei_detections_torch += torch.from_numpy(noise).float()


        data_s = Data(pos=nuclei_detections_torch, y=nuclei_ids) # has keys `pos` and `y` of size N x 3 and N
        
        if self.transform is not None:
            data_s = self.transform(data_s)
        # Now data_s has y = N x 38, pos = N x 3, x = N x 1,      
        
        data = Data(num_nodes=nuclei_detections_torch.size(0))
        
        for key in data_s.keys:
            data['{}'.format(key)] = data_s[key]
        
        
        return data


class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = Linear(1, hidden_channels) # TODO since we pass in 1 each time
        self.edge_encoder = Linear(3, hidden_channels) # TODO --> relative distance between xyz coordinates

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, 38) # TODO --> one-hot encoded output

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training) # N x 64
        
        return self.lin(x) # N x 38 --> soft-maxed for class probabilities # TODO


torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=32)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=1)  # TODO --> default 64
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--save_dir', type=str, default='platynereis')  # TODO
parser.add_argument('--resume_path', type=str, default=None)
parser.add_argument('--save', type=bool, default=True)
args = parser.parse_args()

# path='../data/' python3 platynereis_classification.py
path = os.environ.get('path')
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

transform = T.Compose([
    T.Constant(),
    T.KNNGraph(k=8),  # TODO - k = 8 (default)
    T.Cartesian(),

])

path = osp.join(path)
train_dataset = Dataset(path, type='train', size=None, transform=transform)  # TODO
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
test_dataset = Dataset(path, type='test', size=None, transform=transform)  # TODO
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss() # TODO this includes a softmax



def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = 0
    count = 0
    for data in train_loader:
        
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr) # out has shape N x 38
        loss = criterion(out, data.y.squeeze(1)) # data.y is B i.e. no channels        
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) 
        count +=1
        pbar.update(1)

    pbar.close()

    return total_loss / count


@torch.no_grad()
def test(epoch):
    model.eval()

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')
    total_top_1_accuracy = total_loss = 0
    
    count = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr) # out has shape N x 38
        loss = criterion(out, data.y.squeeze(1))
        accuracy_emb = torch.sum(torch.argmax(F.softmax(out, 1),1) == data.y.squeeze(1))
        total_top_1_accuracy += accuracy_emb.cpu().detach().numpy()/out.shape[0]
        total_loss +=float(loss)
        count +=1
        pbar.update(1)

    pbar.close()
    return total_loss/count, total_top_1_accuracy/count




for epoch in tqdm(range(1, 1001)):
    loss = train(epoch)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f} ...')
    loss, test_acc = test(epoch)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
    
