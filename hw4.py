# %%
import numpy as np
import torch
import torch.nn.functional as F  #激活函数
from torch import nn
from torch import optim  #优化器
from torchvision import transforms as tf
import matplotlib.pyplot as plt
import os
from PIL import Image

# %%
def get_data(root_dir):
  paths=[]
  labels=[]
  for root, dirs, files in os.walk(root_dir):
    for file in files:
      paths.append(os.path.join(root, file))
      index=file.find('_')
      labels.append(file[:index])
  return paths,labels

def get_folder(root_dir):
  folder_name=[]
  for entry in os.listdir(root_dir):
    full_path=os.path.join(root_dir, entry)
    if os.path.isdir(full_path):
      folder_name.append(entry)
  return folder_name

# %%
class CifarDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, transform=None):
    #初始化数据集，包含一系列图片与对应标签
    super().__init__()
    #self.data_path=root_dir
    self.transform=transform
    dataset=get_folder(root_dir)
    path_data, labels=get_data(root_dir)
    self.path=path_data
    img_data=[]
    for i in path_data:
      img=Image.open(i)
      img=img.convert('L')
      img_data.append(img)
      img.close()
    self.data=img_data
    self.label=labels
    if(dataset[0][-4:]=='test'):
      test_data, test_labels=get_data(dataset[0])
      train_data, train_labels=get_data(dataset[1])
    else:
      test_data, test_labels=get_data(dataset[1])
      train_data, train_labels=get_data(dataset[0])
    img_train=[]
    img_test=[]
    for j in train_data:
      img=Image.open(j)
      img=img.convert('L')
      img_train.append(img)
      img.close()
    for k in test_data:
      img=Image.open(k)
      img=img.convert('L')
      img_test.append(img)
      img.close()
    self.train_label=train_labels
    self.train=img_train
    self.test_label=test_labels
    self.test=img_test
    #raise NotImplementedError
  
  def __len__(self):
    #返回数据集大小
    return len(self.data)
    #raise NotImplementedError
  
  def __getitem__(self, index):
    #返回第index个样本
    img=self.data[index]
    label=self.label[index]
    path=self.path[index]
    if(self.transform):
      img=self.transform(img)
    item={'image':img, 'label':label, 'path':path}
    return item
    #raise NotImplementedError
  
  

# %%
def get_mean(dataset):
  expect_sum, var_sum=torch.zeros(3), torch.zeros(3)
  size=len(dataset)
  transform=tf.Compose([tf.ToTensor(), tf.Normalize((0,), (1,))])
  for i in range(size):
    image=Image.open(dataset[i]['path'])
    img_tensor=transform(image)
    expect_sum=expect_sum+torch.mean(img_tensor, dim=[1,2])
    var_sum=var_sum+torch.mean(img_tensor**2, dim=[1,2])
  mean=expect_sum/size
  std=var_sum/size
  return mean, std

# %%
#FCNModel
def get_dataloader(train=True):
  if(train):
    dataset=CifarDataset('..\cifar10\cifar10_'+'train')
  else:
    dataset=CifarDataset('..\cifar10\cifar10_'+'test')
  mean, std=get_mean(dataset)
  t=tf.Compose([tf.ToTensor(), tf.Normalize(mean, std), tf.Grayscale(num_output_channels=1)])
  i=0
  dataset_o=[]
  d=[]
  label_o=[]
  for i in range(len(dataset)):
    img=Image.open(dataset[i]['path'])    
    img=t(img)
    d.append(img)
    x=0
    if(dataset[i]['label']=='airplane'):x=0
    elif(dataset[i]['label']=='automobile'):x=1
    elif(dataset[i]['label']=='bird'):x=2
    elif(dataset[i]['label']=='cat'):x=3
    elif(dataset[i]['label']=='deer'):x=4
    elif(dataset[i]['label']=='dog'):x=5
    elif(dataset[i]['label']=='frog'):x=6
    elif(dataset[i]['label']=='horse'):x=7
    elif(dataset[i]['label']=='ship'):x=8
    else:x=9
    label_o.append(x)
  dataset_o=list(zip(d, label_o))
  if train:
    batch_size=train_batch_size 
  else:
    batch_size=test_batch_size
  dataloader=torch.utils.data.DataLoader(dataset_o, batch_size=batch_size, shuffle=True)
  return dataloader

train_batch_size=128
test_batch_size=64

# %%
class FCNModel(torch.nn.Module):
  def __init__(self):
    super(FCNModel, self).__init__()
    self.fc1=torch.nn.Linear(32*32*1, 128)
    self.fc2=torch.nn.Linear(128, 32)
    self.fc3=torch.nn.Linear(32,10)

  def forward(self, input_data):
    x=input_data.view(-1, 32*32*1)
    x=self.fc1(x)
    x=F.relu(x)
    x=self.fc2(x)
    x=F.relu(x)
    out=F.log_softmax(x, dim=-1)
    return out

# %%
fcn_model=FCNModel()
optimizer=torch.optim.Adam(fcn_model.parameters(), lr=0.001)
train_loss_list=[]
train_count_list=[]
def fcntrain(epoch):
  fcn_model.train(True)
  train_dataloader=get_dataloader(True)
  print("start training")
  for id,sample in enumerate(train_dataloader):
    data, label=sample
    label_t=torch.tensor(label)   
    optimizer.zero_grad()
    out=fcn_model(data)
    loss=F.nll_loss(out, label_t)
    loss.backward()
    optimizer.step()
    if id%200==0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,  id * len(data), len(train_dataloader.dataset),100. * id / len(train_dataloader), loss.item()))
      train_loss_list.append(loss.item())
      train_count_list.append(id*train_batch_size+(epoch-1)*len(train_dataloader))
  print('end training')

# %%
def fcntest():
    test_loss = 0
    correct = 0
    fcn_model.eval()  #设置为评估模式
    test_dataloader = get_dataloader(train=False)  #导入测试数据集
    with torch.no_grad():  #不需要计算梯度
        for data, label in test_dataloader:
            output = fcn_model(data)
            test_loss += F.nll_loss(output, label, reduction='sum').item()  
            pred = output.data.max(1, keepdim=True)[1] #获取最大值的位置
            correct += pred.eq(label.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)  #计算平均损失
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))

# %%
for i in range(5):
    fcntrain(i)
fcntest()

# %%
#cnnModel
class cnnModel(torch.nn.Module):
  def __init__(self):
    super(cnnModel, self).__init__()
    self.conv1=nn.Conv2d(1, 6, 5, padding=2)
    self.conv2=nn.Conv2d(6, 18, 5)
    self.fc1=nn.Linear(18*6*6, 128)
    self.dropout=nn.Dropout(p=0.2)
    self.fc2=nn.Linear(128, 32)
    self.fc3=nn.Linear(32, 10)

  def forward(self, x):
    x=F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x=F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
    x=x.view(-1, self.num_flat_features(x))
    x=F.relu(self.fc1(x))
    x=self.dropout(x)
    x=F.relu(self.fc2(x))
    x=self.fc3(x)
    out=F.log_softmax(x, dim=-1)
    return out
  
  def num_flat_features(self, x):
    size=x.size()[1:]
    num=1
    for s in size:
      num=num*s
    return num
  

# %%
cnn=cnnModel()
optimizer = torch.optim.Adam(cnn.parameters(),lr= 0.001) 
train_loss_list = []
train_count_list = []
train_batch_size = 128
test_batch_size = 64
def cnntrain(epoch):
  cnn.train(True)
  train_dataloader=get_dataloader(True)
  print("start training")
  for id,sample in enumerate(train_dataloader):
    data, label=sample
    label_t=torch.tensor(label)
    optimizer.zero_grad()
    out=cnn(data)
    loss=F.nll_loss(out, label_t)
    loss.backward()
    optimizer.step()
    if id%200==0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,  id * len(data), len(train_dataloader.dataset),100. * id / len(train_dataloader), loss.item()))
      train_loss_list.append(loss.item())
      train_count_list.append(id*train_batch_size+(epoch-1)*len(train_dataloader))
  print("end training")


# %%
def cnntest():
    test_loss = 0
    correct = 0
    cnn.eval()  #设置为评估模式
    test_dataloader = get_dataloader(train=False)  #导入测试数据集
    with torch.no_grad():  #不需要计算梯度
        for data, target in test_dataloader:
            output = cnn(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.data.max(1, keepdim=True)[1] #获取最大值的位置,[batch_size,1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)  #计算平均损失
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))

# %%
for i in range(5):
    cnntrain(i)
cnntest()


