PYTORCH_DIR = '/dbfs/ml/horovod_pytorch'

batch_size = 100
num_epochs = 3
momentum = 0.5
log_interval = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from time import time
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train_one_epoch(model, device, data_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(data_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(data_loader) * len(data),
            100. * batch_idx / len(data_loader), loss.item()))

def save_checkpoint(log_dir, model, optimizer, epoch):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.module.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)
  
def load_checkpoint(log_dir, epoch=num_epochs):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  return torch.load(filepath)

def train(multi_node_dir, learning_rate):
  #### Added imports here ####
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torch.utils.data.distributed import DistributedSampler
  ############################
  
  print("Running distributed training")
  dist.init_process_group("nccl")
  local_rank = int(os.environ["LOCAL_RANK"])

  train_dataset = datasets.MNIST(
    'data',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  
  #### Added Distributed Dataloader ####
  train_sampler = DistributedSampler(dataset=train_dataset)
  data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
  ######################################
  
  model = Net().to(local_rank)
  #### Added Distributed Model ####
  ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
  #################################

  optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=momentum)
  for epoch in range(1, num_epochs + 1):
    train_one_epoch(ddp_model, local_rank, data_loader, optimizer, epoch)
    save_checkpoint(multi_node_dir, ddp_model, optimizer, epoch)
  
  dist.destroy_process_group()

def test(log_dir):
  device = torch.device('cuda')
  loaded_model = Net().to(device)
  
  checkpoint = load_checkpoint(log_dir)
  loaded_model.load_state_dict(checkpoint['model'])
  loaded_model.eval()

  test_dataset = datasets.MNIST(
    'data', 
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  data_loader = torch.utils.data.DataLoader(test_dataset)

  test_loss = 0
  for data, target in data_loader:
      data, target = data.to(device), target.to(device)
      output = loaded_model(data)
      test_loss += F.nll_loss(output, target)
  
  test_loss /= len(data_loader.dataset)
  print("Average test loss: {}".format(test_loss.item()))

def print_env_vars():
  import os
  keys = ["MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "RANK", "WORLD_SIZE"]
  vals = [os.environ[var] for var in keys]
  print(list(zip(keys, vals)))

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("dir", help="where to store the data")
  parser.add_argument("lr", help="learning_rate", default=0.001)
  args = parser.parse_args()
  print("storing the data in: ", args.dir)
  print("learning rate chosen: ", float(args.lr))
  train(args.dir, float(args.lr))