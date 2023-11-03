# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed Training E2E on a Databricks Notebook

# COMMAND ----------

# MAGIC %pip install pytorch_frame

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## MLFlow setup

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/pytorch-distributor'

# We will need these later
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# We manually create the experiment so that we know the id and can send that to the worker nodes when we scale
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

import torch
from torch_frame.datasets import TabularBenchmark
from torch_frame.data import DataLoader
from torch_frame.nn import Trompt
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch.nn.functional as F

seed = 0
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_temp_path = '/tmp/tabular_deeplearning'
dbutils.fs.mkdirs(data_temp_path)
databricks_path = f'/dbfs{data_temp_path}'

# COMMAND ----------

batch_size = 12
channels = 128
num_prompts = 128
num_layers = 6
lr = 0.001
epochs = 10

dataset = TabularBenchmark(root=databricks_path, name='california')
dataset.materialize()

assert dataset.task_type.is_classification
dataset = dataset.shuffle()

train_dataset, val_dataset, test_dataset = dataset[:0.7], dataset[
    0.7:0.79], dataset[0.79:]

train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

# COMMAND ----------

model = Trompt(
    channels=channels,
    out_channels=dataset.num_classes,
    num_prompts=num_prompts,
    num_layers=num_layers,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

# COMMAND ----------

def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        # [batch_size, num_layers, num_classes]
        out = model(tf)
        num_layers = out.size(1)
        # [batch_size * num_layers, num_classes]
        pred = out.view(-1, dataset.num_classes)
        y = tf.y.repeat_interleave(num_layers)
        # Layer-wise logit loss
        loss = F.cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count

# COMMAND ----------

best_val_acc = 0
best_test_acc = 0
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    #train_acc = test(train_loader)
    #val_acc = test(val_loader)
    #test_acc = test(test_loader)
    #if best_val_acc < val_acc:
    #    best_val_acc = val_acc
    #    best_test_acc = test_acc
    #print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
    #      f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Train Loss: {train_loss}')

    lr_scheduler.step()

#print(f'Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')

# COMMAND ----------

def wrapped_train():
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
    #train_acc = test(train_loader)
    #val_acc = test(val_loader)
    #test_acc = test(test_loader)
    #if best_val_acc < val_acc:
    #    best_val_acc = val_acc
    #    best_test_acc = test_acc
    #print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
    #      f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Train Loss: {train_loss}')

        lr_scheduler.step()

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

dist = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True).run(wrapped_train) 

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

dist = TorchDistributor(num_processes=2, local_mode=False, use_gpu=True).run(wrapped_train) 

# COMMAND ----------