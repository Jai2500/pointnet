import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric

class PointNetModule(torch.nn.Module):
  def __init__(self, fps_ratio, query, nn):
    super(PointNetModule, self).__init__()
    self.fps_ratio = fps_ratio
    self.query = query
    self.conv = torch_geometric.nn.PointConv(nn)

  def forward(self, x, pos, batch):
    idx = torch_geometric.nn.fps(pos, batch, ratio=self.fps_ratio)
    row, col = torch_geometric.nn.radius(pos, pos[idx], self.query, batch, 
                                         batch[idx], max_num_neighbors=64)
    edge_idx = torch.stack([col, row], dim=0)
    x = self.conv(x, (pos, pos[idx]), edge_idx)
    pos = pos[idx]
    batch = batch[idx]
    
    return x, pos, batch


class PointNetPool(torch.nn.Module):
  def __init__(self, nn):
    super(PointNetPool, self).__init__()
    self.nn = nn

  def forward(self, x, pos, batch):
    x = self.nn(torch.cat([x, pos], dim=1))
    x = torch_geometric.nn.global_max_pool(x, batch)
    return x


def mlp(channels, batch_norm=True):
    return torch.nn.Sequential(*[
        torch.nn.Sequential(torch.nn.Linear(channels[i - 1], channels[i]), torch.nn.ReLU(), torch.nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = PointNetModule(0.5, 0.2, mlp([3, 64, 64, 128]))
    self.conv2 = PointNetModule(0.25, 0.4, mlp([128 + 3, 128, 128, 256]))
    self.pool = PointNetPool(mlp([256 + 3, 256, 512, 1024]))

    self.lin1 = torch.nn.Linear(1024, 512)
    self.lin2 = torch.nn.Linear(512, 256)
    self.lin3 = torch.nn.Linear(256, 10)

  def forward(self, data):
    out = (data.x, data.pos, data.batch)
    out = self.conv1(*out)
    out = self.conv2(*out)
    out = self.pool(*out)

    x = F.relu(self.lin1(out))
    x = F.dropout(x, p=0.5, training=self.training)
    x = F.relu(self.lin2(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin3(x)

    return F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == "__main__":
    path = osp.join('.', 'data/ModelNet10')
    
    pre_transform, transform = torch_geometric.transforms.NormalizeScale(), torch_geometric.transforms.SamplePoints(1024)

    train_dataset = torch_geometric.datasets.ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = torch_geometric.datasets.ModelNet(path, '10', False, pre_transform, transform)

    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
      train(epoch, optimizer)
      test_acc = test(test_loader)
      print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))