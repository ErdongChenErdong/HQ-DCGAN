import time
import torch
from torch.utils.data import dataloader, Dataset
from torch.optim import SGD
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('../ECG_CNN')


def train():

    # 初始化网络
    gen = PatchQuantumGenerator(n_generators).to(device)
    dis = Discriminator().to(device)

    # 初始化优化器
    # d_optim = RMSprop(dis.parameters(), lr=0.001)
    # g_optim = RMSprop(gen.parameters(), lr=0.01)
    d_optim = SGD(dis.parameters(), lr=0.01)
    g_optim = SGD(gen.parameters(), lr=0.3)
    loss_fn = torch.nn.BCELoss()

    # 记录生成器生成的数据的均值和方差
    G_mean = []
    G_std = []

    # 判别器训练轮数
    d_epoch = 1

    # D和G的损失值
    D_loss = []
    G_loss = []

    # plt.ion()
    # for epoch in range(start_epoch+1, 45000)
    for epoch in range(Epoch):
        d_epoch_loss = 0
        g_epoch_loss = 0
        count = len(dataloader)
        gen.train()

        for step, (data, _) in enumerate(dataloader):
            data = data.to(device)
            size = data.size(0)
            d_optim.zero_grad()
            real_output = dis(data)
            d_real_loss = loss_fn(real_output, torch.ones_like(real_output))
            random_noise = torch.rand(size, n_qubits, device=device) * math.pi / 2
            gen_data = gen(random_noise)
            fake_output = dis(gen_data.detach())
            d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            g_optim.zero_grad()
            fake_output = dis(gen_data)
            g_loss = loss_fn(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            g_optim.step()
            G_mean.append(fake_output.mean().item())
            G_std.append(fake_output.std().item())

            with torch.no_grad():
                d_epoch_loss += d_loss
                g_epoch_loss += g_loss
            if epoch == 1500 or epoch == 1501:
                global Fake_data
                if epoch == 1500 and step == 0:
                    Fake_data = gen_data
                    print(Fake_data.shape)
                else:
                    Fake_data = torch.cat((Fake_data, gen_data), 0)
                    # print(gen_data.shape)

        with torch.no_grad():
            d_epoch_loss /= count
            g_epoch_loss /= count
            D_loss.append(d_epoch_loss)
            G_loss.append(g_epoch_loss)


            # print('Epoch:', epoch)
        if epoch != 0 and epoch % 10 == 0:
            print("Epoch: {} | Loss_D: {} | Loss_G: {} | Time: {}".format(epoch, G_loss[-1], D_loss[-1], time.strftime('%H:%M:%S')))


if __name__ == '__main__':
    train()