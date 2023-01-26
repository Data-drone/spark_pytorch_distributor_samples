# Databricks notebook source
# MAGIC %md
# MAGIC ## Distributed Training E2E on a Databricks Notebook
# MAGIC 
# MAGIC Distributed training on PyTorch is often done by creating a file (`train.py`) and using the `torchrun` CLI to run distributed training using that file. Nevertheless, Databricks offers a method of doing distributed training directly on the notebook. You can define the `train()` function within a notebook and use the `TorchDistributor` API to train the model across the workers.
# MAGIC 
# MAGIC This notebook will illustrates how to develop interactively within a notebook. Especially with larger deep learning projects we also recommend leveraging the `%run` command in order to split up your code into manageable chunks.
# MAGIC 
# MAGIC We will firstly code up a simple single GPU model training on the classic MNIST dataset then  adapt that code for distributed training before showing how the TorchDistributor can be leveraged to help you scale up the model training across multiple nodes. 
# MAGIC 
# MAGIC ## Requirements
# MAGIC - Databricks Runtime ML 13.0 and above
# MAGIC - (Recommended) GPU instances
# MAGIC - See: 
# MAGIC   - GPU on AWS: https://docs.databricks.com/clusters/gpu.html
# MAGIC   - GPU on Azure: https://learn.microsoft.com/en-gb/azure/databricks/clusters/gpu
# MAGIC   - GPU on GCP: https://docs.gcp.databricks.com/clusters/gpu.html

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### MLFlow setup
# MAGIC 
# MAGIC MLflow is a tool to support the tracking of machine learning experiments and logging of models
# MAGIC 
# MAGIC ***NOTE*** The Mlflow PyTorch Autologging APIs are designed for PyTorch Lightning and won't work with Native PyTorch

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/pytorch-distributor'

# We will need these later
db_host = "https://e2-demo-tokyo.cloud.databricks.com/"  # CHANGE THIS!
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# We manually create the experiment so that we know the id and can send that to the worker nodes when we scale
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md ## Defining Useful Functions
# MAGIC 
# MAGIC The following cell contains code that describes the model, the train function, and the testing function. All of which is designed to run locally. We will then introduce changes that are needed to move training from the local setting to a distributed setting.
# MAGIC 
# MAGIC All the torch code leverages standard PyTorch APIs, there are no custom libraries required or alterations in the way the code is written. In these notebooks, we will be focusing on how to scale your training with `Torchdistributor` and will not go through the modelling code. 

# COMMAND ----------

import torch
NUM_WORKERS = 2
NUM_GPUS_PER_NODE = torch.cuda.device_count()

# COMMAND ----------

PYTORCH_DIR = '/dbfs/ml/pytorch'

batch_size = 100
num_epochs = 3
momentum = 0.5
log_interval = 100
learning_rate = 0.001

import torch
import torch.nn as nn
import torch.nn.functional as F

# Our Model definition
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
            
            mlflow.log_metric('train_loss', loss.item())

def save_checkpoint(log_dir, model, optimizer, epoch):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)
  
def load_checkpoint(log_dir, epoch=num_epochs):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  return torch.load(filepath)

def create_log_dir():
  log_dir = os.path.join(PYTORCH_DIR, str(time()))
  os.makedirs(log_dir)
  return log_dir

import torch.optim as optim
from torchvision import datasets, transforms
from time import time
import os

base_log_dir = create_log_dir()
print("Log directory:", base_log_dir)

def train():
  device = torch.device('cuda')

  train_parameters = {'batch_size': batch_size, 'epochs': num_epochs}
  mlflow.log_params(train_parameters)
  
  train_dataset = datasets.MNIST(
    'data', 
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  model = Net().to(device)

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, data_loader, optimizer, epoch)
    save_checkpoint(base_log_dir, model, optimizer, epoch)
    
def test(log_dir):
  device = torch.device('cuda')
  loaded_model = Net().to(device)
  scripted_model = torch.jit.script(loaded_model)
  
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
  
  mlflow.log_metric('test_loss', test_loss.item())
  
  mlflow.pytorch.log_model(scripted_model, "model")
  

# COMMAND ----------

# MAGIC %md ### Training the Code Locally
# MAGIC 
# MAGIC To test that this runs correctly, we will trigger a train and test iteration using the functions we defined above.

# COMMAND ----------

with mlflow.start_run():
  
  mlflow.log_param('run_type', 'local')
  train()
  test(base_log_dir)
  

# COMMAND ----------

# MAGIC %md ## Distributed Setup
# MAGIC 
# MAGIC To get the code ready for distributed training, we will need to make some changes. Firstly, we will need to set up some environment variables that are required for running the distributed `train()` function (repeatedly) locally:
# MAGIC 
# MAGIC ```
# MAGIC os.environ["MASTER_ADDR"] = "localhost"
# MAGIC os.environ["MASTER_PORT"] = "9340"
# MAGIC os.environ["RANK"] = str(0)
# MAGIC os.environ["LOCAL_RANK"] = str(0)
# MAGIC os.environ["WORLD_SIZE"] = str(1)
# MAGIC ```
# MAGIC 
# MAGIC When you wrap the single-node code in the `train()` function, Databricks recommends you to include all the import statements inside the train() function to avoid library pickling issues.
# MAGIC 
# MAGIC Everything else is what is normally required for getting distributed training to work within PyTorch.
# MAGIC - Calling `dist.init_process_group("nccl")` at the beginning of `train()`
# MAGIC - Calling `dist.destroy_process_group()` at the end of `train()`
# MAGIC - Setting `local_rank = int(os.environ["LOCAL_RANK"])`
# MAGIC - Adding a `DistributedSampler` to the `DataLoader`
# MAGIC - Wrapping the model with a `DDP(model)`
# MAGIC - For more information, view https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html

# COMMAND ----------

single_node_single_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_single_gpu_dir)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9340"
os.environ["RANK"] = str(0)
os.environ["LOCAL_RANK"] = str(0)
os.environ["WORLD_SIZE"] = str(1)

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
      
      if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('train_loss', loss.item())

def save_checkpoint(log_dir, model, optimizer, epoch):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.module.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)

# For distributed training we will merge the train and test steps into 1 main function
def main_fn(directory):
  
  #### Added imports here ####
  import mlflow
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torch.utils.data.distributed import DistributedSampler
  
  ############################
  ##### Setting up MLflow ####
  # We need to do this so that different processes that will 
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  # We set the experiment details here
  experiment = mlflow.set_experiment(experiment_path)
  
  ############################
  
  ############################
  print("Running distributed training")
  dist.init_process_group("nccl")
#   dist.init_process_group(backend="nccl", world_size=2)
  
  local_rank = int(os.environ["LOCAL_RANK"])
  global_rank = int(os.environ["RANK"])
  
  if global_rank == 0:
    train_parameters = {'batch_size': batch_size, 'epochs': num_epochs, 'trainer': 'TorchDistributor'}
    mlflow.log_params(train_parameters)
  
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
    
    if global_rank == 0: 
      save_checkpoint(directory, ddp_model, optimizer, epoch)
  
  # save out the model for test
  if global_rank == 0:
    mlflow.pytorch.log_model(ddp_model, "model")
    
    ddp_model.eval()
    test_dataset = datasets.MNIST(
      'data', 
      train=False,
      download=True,
      transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    data_loader = torch.utils.data.DataLoader(test_dataset)    

    test_loss = 0
    for data, target in data_loader:
      device = torch.device('cuda')
      data, target = data.to(device), target.to(device)
      output = ddp_model(data)
      test_loss += F.nll_loss(output, target)
          
    test_loss /= len(data_loader.dataset)
    print("Average test loss: {}".format(test_loss.item()))
    
    mlflow.log_metric('test_loss', test_loss.item())

    
  dist.destroy_process_group()
  
  return "finished" # can return any picklable object

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Testing without Distributor
# MAGIC 
# MAGIC This is just to validate that there is nothing wrong with our train loop.
# MAGIC It is training just on a single GPU.

# COMMAND ----------

# single node distributed run
# we want to just quickly test that the whole process is working
with mlflow.start_run():
  
  mlflow.log_param('run_type', 'test_dist_code')
  main_fn(single_node_single_gpu_dir)
  

# COMMAND ----------

# MAGIC %md ### Single Node Multi GPU training
# MAGIC 
# MAGIC PyTorch provides a [roundabout way](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html) for doing single node multi-GPU (SNMG) training. Databricks provides a more streamlined solution that allows you to move from SNMG to multi node training seamlessly. To do SNMG training on Databricks, you would just need to invoke the `TorchDistributor` API and set `num_processes` equal to the number of available GPUs on the driver node that you want to use and set `local_mode=True`.

# COMMAND ----------

single_node_multi_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_multi_gpu_dir)

from spark_pytorch_distributor.distributor import TorchDistributor

output = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True).run(main_fn, single_node_multi_gpu_dir)
  #test(single_node_multi_gpu_dir)

# COMMAND ----------

# MAGIC %md ### Multi-Node Training
# MAGIC 
# MAGIC To move from SNMG training to multi-node training, you just change `num_processes` to the number of GPUs that you want to use across all clusters. For this example, we will use all available GPUs (`NUM_GPUS_PER_NODE * NUM_WORKERS`). You also change `local_mode` to `False`. Additionally, to configure how many GPUs to use for each spark task that runs the train function, `set spark.task.resource.gpu.amount <num_gpus_per_task>` in the Spark Config cell on the cluster page before creating the cluster.

# COMMAND ----------

single_node_multi_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_multi_gpu_dir)

# NUM_GPUS_PER_NODE * NUM_WORKERS
output_dist = TorchDistributor(num_processes=2, local_mode=False, use_gpu=True).run(main_fn, single_node_multi_gpu_dir)


# COMMAND ----------


