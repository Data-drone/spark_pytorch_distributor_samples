# Databricks notebook source
# DBTITLE 1,Note that there is an issue with mlflow-skinny that will require waiting on PyTorch Lightning 1.9.1
#%pip install mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC # Distributed Training using PyTorch Lightning
# MAGIC 
# MAGIC [Pytorch Lightning](https://www.pytorchlightning.ai/) provides a simplified API to train machine learning models on PyTorch without any of the boilerplate code. Coupled with PySpark's [`TorchDistributor`](https://github.com/apache/spark/blob/master/python/pyspark/ml/torch/distributor.py), you can launch distributed training tasks using a Spark job in barrier mode. Users only need to provide a `train()` function that runs the single-node training code on a GPU or worker node and the package will handle all the configurations for you.
# MAGIC 
# MAGIC Pytorch Lightning does not come prebundled with databricks and will need to be installed. It is possible to install it as a notebook library if just testing on single node but it must be installed as a cluster library in order to use it on a cluster.
# MAGIC 
# MAGIC See:
# MAGIC - AWS: [Notebook Library](https://docs.databricks.com/libraries/notebooks-python-libraries.html) / [Cluster Library](https://docs.databricks.com/libraries/cluster-libraries.html)
# MAGIC - Azure: [Notebook Library](https://learn.microsoft.com/en-gb/azure/databricks/libraries/notebooks-python-libraries) / [Cluster Library](https://learn.microsoft.com/en-gb/azure/databricks/libraries/cluster-libraries)
# MAGIC - GCP: [Notebook Library](https://docs.gcp.databricks.com/libraries/notebooks-python-libraries.html) /  [Cluster Library](https://docs.gcp.databricks.com/libraries/cluster-libraries.html)
# MAGIC 
# MAGIC 
# MAGIC ## Requirements
# MAGIC - Databricks Runtime ML 13.0 and above
# MAGIC - (Recommended) GPU instances

# COMMAND ----------

import torch
from torch import optim, nn, utils, Tensor
from torchvision import datasets, transforms
import pytorch_lightning as pl
from spark_pytorch_distributor.distributor import TorchDistributor
import mlflow
import os

NUM_WORKERS = 2
# Assume the driver node and worker nodes have the same instance type.
NUM_GPUS_PER_WORKER = torch.cuda.device_count()
USE_GPU = NUM_GPUS_PER_WORKER > 0

username = spark.sql("SELECT current_user()").first()['current_user()']

experiment_path = f'/Users/{username}/pytorch-distributor'

# We will need these later
db_host = "https://e2-demo-tokyo.cloud.databricks.com/"  # CHANGE THIS!
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# We manually create the experiment so that we know the id and can send that to the worker nodes when we scale
experiment = mlflow.set_experiment(experiment_path)

data_path = f'/dbfs/Users/{username}/data/mnist' # change this to location on DBFS
log_path = f"/dbfs/Users/{username}/pl_training_logger" # change this to location on DBFS

# COMMAND ----------

# MAGIC %md ## Setting up the Model
# MAGIC 
# MAGIC We will be creating an AutoEncoder using the [pl.LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) API. 

# COMMAND ----------

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    ### TODO ###
    ### val_step ###
    
    # Essentially a replica of training_step
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Setting up the Data Module

# COMMAND ----------

class MnistDataModule(pl.LightningDataModule):
  def __init__(self, data_dir:str, batch_size:int=32):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    
    self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])
    
  def setup(self, stage:str):
    self.mnist_test = datasets.MNIST(self.data_dir, download=True, train=False, transform=self.transform)
    self.mnist_predict = datasets.MNIST(self.data_dir, download=True, train=False, transform=self.transform)
    mnist_full = datasets.MNIST(self.data_dir, download=True, train=True, transform=self.transform)
    self.mnist_train, self.mnist_val = utils.data.random_split(mnist_full, [55000, 5000])

  def train_dataloader(self):
    return utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size)

  def val_dataloader(self):
    return utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)

  def test_dataloader(self):
    return utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)

  def predict_dataloader(self):
    return utils.data.DataLoader(self.mnist_predict, batch_size=self.batch_size)

# COMMAND ----------

# MAGIC %md ## Creating the Training Function
# MAGIC 
# MAGIC Note that the TorchDistributor API has support for single node multi-GPU training as well as multi-node training. The following `pl_train` function takes the parameters `num_tasks` and `num_proc_per_task`.
# MAGIC 
# MAGIC For additional clarity:
# MAGIC - `num_tasks` (which sets `pl.Trainer(num_nodes=num_tasks, **kwargs)`) is the number of **Spark Tasks** you want for distributed training.
# MAGIC - `num_proc_per_task` (which sets `pl.Trainer(devices=num_proc_per_task, **kwargs)`) is the number of devices/GPUs you want per **Spark task** for distributed training.
# MAGIC 
# MAGIC If you are running single node multi-GPU training on the driver, set `num_tasks` to 1 and `num_proc_per_task` to the number of GPUs that you want to use on the driver.
# MAGIC 
# MAGIC If you are running multi-node training, set `num_tasks` to the number of spark tasks you want to use and `num_proc_per_task` to the value of `spark.task.resource.gpu.amount` (which is usually 1).
# MAGIC 
# MAGIC Therefore, the total number of GPUs used is `num_tasks * num_proc_per_task`

# COMMAND ----------

def main_training_loop(num_tasks, num_proc_per_task):

  """
  
  Main train and test loop
  
  """
  # add imports inside pl_train for pickling to work
  from torch import optim, nn, utils, Tensor
  from torchvision import datasets, transforms
  import pytorch_lightning as pl
  import mlflow
  
  ############################
  ##### Setting up MLflow ####
  # We need to do this so that different processes that will be able to find mlflow
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  
  epochs = 14
  batch_size = 32
  
  mlf_logger = pl.loggers.MLFlowLogger(experiment_name=experiment_path)
  
  # define any number of nn.Modules (or use your current ones)
  encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
  decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

  # init the autoencoder
  autoencoder = LitAutoEncoder(encoder, decoder)

  datamodule = MnistDataModule(data_dir = data_path, batch_size = batch_size)
  
  # train the model
  if num_tasks == 1 and num_proc_per_task == 1:
    strategy = None
  else:
    strategy = "ddp"
  trainer = pl.Trainer(accelerator='gpu', devices=num_proc_per_task, num_nodes=num_tasks, 
                    strategy=strategy, default_root_dir=log_path, logger=mlf_logger,
                    limit_train_batches=1000, max_epochs=epochs)
   
  trainer.fit(model=autoencoder, datamodule=datamodule)
  
  trainer.test(model=autoencoder, datamodule=datamodule)
  
  
  
  return autoencoder, trainer.checkpoint_callback.best_model_path
  
  

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC git config --global user.name "Data-drone"
# MAGIC git config --global user.email "bpl.law@gmail.com"

# COMMAND ----------

# MAGIC %md ## Model Training (and Testing)

# COMMAND ----------

# MAGIC %md ### Train the model locally with 1 GPU
# MAGIC 
# MAGIC Note that `nnodes` = 1 and `nproc_per_node` = 1.

# COMMAND ----------

NUM_TASKS = 1
NUM_PROC_PER_TASK = 1

model, ckpt_path = main_training_loop(NUM_TASKS, NUM_PROC_PER_TASK)

# COMMAND ----------

# MAGIC %md ## Single Node Multi GPU Setup
# MAGIC 
# MAGIC For the distributor API, you want to set `num_processes` to the total amount of GPUs that you plan on using. For single node multi-gpu, this is limited by the number of GPUs available on the driver node.
# MAGIC 
# MAGIC As mentioned before, single node multi gpu (with `NUM_PROC` GPUs) setup involves setting `trainer = pl.Trainer(accelerator='gpu', devices=NUM_PROC, num_nodes=1, **kwargs)`

# COMMAND ----------

NUM_TASKS = 1
NUM_PROC_PER_TASK = NUM_GPUS_PER_WORKER
NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK

(model, ckpt_path) = TorchDistributor(num_processes=NUM_PROC, local_mode=True, use_gpu=USE_GPU).run(main_training_loop, NUM_TASKS, NUM_PROC_PER_TASK) 

# COMMAND ----------

# MAGIC %md ## Multi Node Setup
# MAGIC 
# MAGIC For the distributor API, you want to set `num_processes` to the total amount of GPUs that you plan on using. For multi-node, this will be equal to `num_spark_tasks * num_gpus_per_spark_task`. Additioanlly, note that `num_gpus_per_spark_task` usually equals 1 unless you configure that value specifically.
# MAGIC 
# MAGIC Note that multi node (with `num_proc` GPUs) setup involves setting `trainer = pl.Trainer(accelerator='gpu', devices=1, num_nodes=num_proc, **kwargs)`

# COMMAND ----------

NUM_TASKS = NUM_WORKERS * NUM_GPUS_PER_WORKER
NUM_PROC_PER_TASK = 1
NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK

## temp
NUM_PROC = 2

(model, ckpt_path) = TorchDistributor(num_processes=NUM_PROC, local_mode=False, use_gpu=True).run(main_training_loop, NUM_TASKS, NUM_PROC_PER_TASK)
