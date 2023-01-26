# Databricks notebook source
# MAGIC %md ## Distributed Training on a PyTorch File
# MAGIC 
# MAGIC Distributed training on PyTorch is often done by creating a file (`train.py`) and using the `torchrun` CLI to run distributed training using that file. Databricks streamlines that process by allowing you to import a file (or even a repository) and use a Databricks notebook to start distributed training on that file using the TorchDistributor API. The example file that we will use in this example is the file: `/Workspace/Repos/rithwik.ediga@databricks.com/spark_pytorch_distributor_8/test.py`.
# MAGIC 
# MAGIC This file is laid out similar to other solutions that use `torchrun` under the hood for distributed training.
# MAGIC 
# MAGIC ## Requirements
# MAGIC - Databricks Runtime ML 13.0 and above
# MAGIC - (Recommended) GPU instances

# COMMAND ----------

import os
from time import time
import torch

PYTORCH_DIR = '/dbfs/ml/pytorch_file_example'

def create_log_dir():
  log_dir = os.path.join(PYTORCH_DIR, str(time()))
  os.makedirs(log_dir)
  return log_dir

single_node_multi_gpu_dir = create_log_dir()
multi_node_dir = create_log_dir()

NUM_WORKERS = 2
NUM_GPUS_PER_NODE = torch.cuda.device_count()

# COMMAND ----------

# MAGIC %md ### Utility Functions
# MAGIC 
# MAGIC The following functions are required for testing the model that was saved in either `single_node_multi_gpu_dir` or `multi_node_dir`.

# COMMAND ----------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

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

def load_checkpoint(log_dir):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=3)
  return torch.load(filepath)

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

# COMMAND ----------

# MAGIC %md ### Local Mode (Training on the Driver)
# MAGIC 
# MAGIC This first cell showcases how to run single node multi GPU setup for distributed training using a file. Internally, the file calls `torch.distributed.init_process_group()` and `torch.distributed.destroy_process_group()`, both of which are functions that are expected for distributed training on PyTorch. To learn more about how the file is set up, view PyTorch's documentation: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# MAGIC 
# MAGIC To configure how many GPUs to use in total for this run, pass `num_processes=TOTAL_NUM_GPUS` to the Distributor. For local mode, this is limited to the number of GPUs available on the driver node.

# COMMAND ----------

## Setting up the right filepath

username = spark.sql("SELECT current_user()").first()['current_user()']
username

repo_path = f'/Workspace/Repos/{username}/spark_pytorch_distributor_samples/Scripts/Basic_MNIST/train.py'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ***Note*** You may get an error the first time you run the following command about `OSError: [Errno 5] Input/output error` This is due to the dataset download process. When you rerun it should run fine

# COMMAND ----------

from spark_pytorch_distributor.distributor import TorchDistributor

trainer = TorchDistributor(num_processes=NUM_GPUS_PER_NODE, local_mode=True, use_gpu=True).run(repo_path, single_node_multi_gpu_dir, "0.01")

# COMMAND ----------

test(single_node_multi_gpu_dir)

# COMMAND ----------

# MAGIC %md ### Multi Node Distributed Training
# MAGIC 
# MAGIC This mode of training allows you to use all possible GPUs on your cluster. You simply need to set `num_processes=TOTAL_NUM_GPUS` and the TorchDistributor will handle the scheduling of GPUs under the hood.
# MAGIC 
# MAGIC To configure how many GPUs to use for each spark task that runs the train function, set `spark.task.resource.gpu.amount <num_gpus_per_task>` in the Spark Config cell on the cluster page before creating the cluster. Normally, `spark.task.resource.gpu.amount` is set to 1, so there will be `TOTAL_NUM_GPUS` processes that are run concurrently for training.
# MAGIC 
# MAGIC Args for frameworks like Argparse can be added after the file_path as comma separated variables

# COMMAND ----------

TOTAL_NUM_GPUS = NUM_GPUS_PER_NODE * NUM_WORKERS

TorchDistributor(num_processes=2, local_mode=False, use_gpu=True).run(repo_path, multi_node_dir, "0.01")

# COMMAND ----------

test(multi_node_dir)

# COMMAND ----------


