import torch
from torch import nn , optim
from torch.utils.data import (Dataset,TensorDataset, DataLoader)
import tqdm
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.datasets import ImageFolder



class DownSizedPairImageFolder(ImageFolder):
    def __init__(self, root, transform=None, large_size=128, small_size=32, **kwds):
        super().__init__(root, transform=transform, **kwds)
        self.large_resizer = transforms.Resize(large_size)
        self.small_resizer = transforms.Resize(small_size)
        
    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        
        #大きな画像をリサイズ
        large_img = self.large_resizer(img)
        #小さな画像をリサイズ
        small_img = self.small_resizer(img)
        
        #大きな画像をTensorに変換
        large_img = self.transform(large_img)
        #小さな画像をTensorに変換
        small_img = self.transform(small_img)
        
        return small_img, large_img


train_data = DownSizedPairImageFolder("lfw-deepfunneled/train/", transform=transforms.ToTensor())
test_data = DownSizedPairImageFolder("lfw-deepfunneled/test/", transform=transforms.ToTensor())

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)


net= nn.Sequential(
    nn.Conv2d(3,256,4,stride=2,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.Conv2d(256,512,4,stride=2,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(512),
    nn.ConvTranspose2d(512,256,4,stride=2,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.ConvTranspose2d(256,128,4,stride=2,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.ConvTranspose2d(64,3,4,stride=2,padding=1),
)

import math
def psnr(mse, max_v=1.0):
    return 10*math.log10(max_v**2/mse)

def eval_net(net, data_loader, device="cpu"):
    net.eval()
    ys=[]
    ypreds=[]
    for x,y in data_loader:
        x=x.to(device)
        y=y.to(device)
        with torch.no_grad():
            y_pred = net(x)
        ys.append(y)
        ypreds.append(y_pred)
    ys=torch.cat(ys)
    ypreds=torch.cat(ypreds)
    ypreds = ypreds.view(ypreds.size(0), ys.size(1), ys.size(2), ys.size(3))  # Reshape ypreds to match ys dimensions
    score=nn.functional.mse_loss(ypreds,ys).item()
    return score





def train_net(net, train_loader, test_loader, optimizer_cls=optim.Adam, loss_fn=nn.MSELoss(), n_iter=10, device="cpu"):
    train_losses=[]
    train_acc=[]
    val_acc=[]
    optimizer=optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss=0.0
        net.train()
        n=0
        score=0
        for i,(xx,yy) in tqdm.tqdm(enumerate(train_loader),total=len(train_loader)):
            xx=xx.to(device)
            yy=yy.to(device)
            y_pred=net(xx)
            loss=loss_fn(y_pred,yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            n+=len(xx)
        train_losses.append(running_loss/i)
        val_acc.append(eval_net(net,test_loader,device))
        print(epoch,train_losses[-1],psnr(train_losses[-1]),psnr(val_acc[-1]),flush=True)

net.to("cuda:0")
train_net(net,train_loader,test_loader,n_iter=20,device="cuda:0")


# 画像を拡大してオリジナルと比較する
from torchvision.utils import save_image

random_test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
it = iter(random_test_loader)
x, y = next(it)

bl_recon = torch.nn.functional.upsample(x, 128, mode='bilinear', align_corners=True)
yp = net(x.to("cuda:0")).to("cpu")
save_image(torch.cat([y,bl_recon,yp], 0), "cnn_upscale1.jpg", nrow=4)


net.to("cpu")
param=net.state_dict()
torch.save(param, "cnn_upscale1.param", pickle_protocol=4)

