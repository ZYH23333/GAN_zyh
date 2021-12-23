import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Generator, Discriminator

# 1.模型搭建
# 2.损失函数
# 3.优化器
# 4.训练框架搭建


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--G_layer_cnt', type=int, default=4)
    parser.add_argument('--D_layer_cnt', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    epochs = args.epochs

    transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    datasets = datasets.MNIST(root=r'F:\DataSets',download=True, transform=transforms)
    train_data = DataLoader(dataset=datasets, batch_size=8, shuffle=True)


    G_model = Generator(
        layer_cnt=args.G_layer_cnt
    )

    D_model = Discriminator(

    )

    for epoch in range(epochs):
        pbar = tqdm(train_data)
        for x, y in pbar:
            p_g = G_model()
            # plt.imshow(p_g.detach())
            # plt.show()
            res = D_model(p_g)
            print(res)