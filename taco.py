import torch
from torch import nn , optim
from torch.utils.data import (Dataset,TensorDataset, DataLoader)
import tqdm
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.datasets import ImageFolder


#ImageFolder関数を利用してDatasetに変換する
train_imges = ImageFolder("taco_and_burrito/train/", transform=transforms.Compose([
    #ImageNetは224×224で学習されているので、それに合わせる
    transforms.RandomCrop(224),
    transforms.ToTensor()
]))

test_imges = ImageFolder("taco_and_burrito/test/", transform=transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor()
]))

train_loader = DataLoader(train_imges, batch_size=32, shuffle=True)
test_loader = DataLoader(test_imges, batch_size=32, shuffle=False)

#train_imgesの中身を確認
print(train_imges.__getitem__(0)[0].size())

print(train_imges.class_to_idx)


from torchvision import models

#resnet18をロード
net = models.resnet18(pretrained=True)

#全てのパラメータを微分対象外にする
for p in net.parameters():
    p.requires_grad=False


#resnet18のネットワーク構成を確認
print(net)

#resnet18の最後の線形層の名前はfc　元々1000分類用になっている
#最後の線形層を付け替える　＋2分類用に変更
fc_input_dim = net.fc.in_features
net.fc = nn.Linear(fc_input_dim, 2)

def eval_net(net, data_loader, device="cpu"):
    net.eval()
    ys=[]
    ypreds=[]
    for x,y in data_loader:
        x=x.to(device)
        y=y.to(device)
        with torch.no_grad():
            _,y_pred=torch.max(net(x),1)
        ys.append(y)
        ypreds.append(y_pred)
    ys=torch.cat(ys)
    ypreds=torch.cat(ypreds)
    acc=(ys==ypreds).float().sum()/len(ys)
    return acc.item()

#only_fc=Trueの場合は、最後の線形層のパラメータのみを更新する
def train_net(net,train_loder,test_loder,only_fc=True,optimizer_cls=optim.Adam,loss_fn=nn.CrossEntropyLoss(),n_iter=10,device="cpu"):
    train_losses=[]
    train_acc=[]
    val_acc=[]
    if only_fc:
        #only_fc=Trueの場合は、最後の線形層のパラメータのみを更新する
        optimizer=optimizer_cls(net.fc.parameters())
    else:
        optimizer=optimizer_cls(net.parameters())
    

    for epoch in range(n_iter):
        running_loss=0.0
        net.train()
        n=0
        n_acc=0
        for i, (xx,yy) in tqdm.tqdm(enumerate(train_loder),total=len(train_loder)):
            xx=xx.to(device)
            yy=yy.to(device)
            h=net(xx)
            loss=loss_fn(h,yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            n+=len(xx)
            _,y_pred=h.max(1)
            n_acc+=(y_pred==yy).sum().item()
        train_losses.append(running_loss/i)
        train_acc.append(n_acc/n)
        val_acc.append(eval_net(net,test_loder,device))
        print(epoch, train_losses[-1],train_acc[-1],val_acc[-1],flush=True)


net.to("cuda:0")
train_net(net,train_loader,test_loader,n_iter=20,device="cuda:0")

net.to("cpu")
params = net.state_dict()
torch.save(params, "taco_burrito_fine_tuning.param")