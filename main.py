import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from VAE import VAE
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import numpy as np
import scanpy as sc
import pandas as pd
from umap import UMAP
import kmeans

# 设置模型运行的设备
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# 设置默认参数
parser = argparse.ArgumentParser(description="Variational Auto-Encoder MNIST Example")
parser.add_argument('--result_dir', type=str, default='./VAEResult', metavar='DIR', help='output directory')
parser.add_argument('--save_dir', type=str, default='./checkPointv1', metavar='N', help='model saving directory')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size for training(default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train(default: 200)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed(default: 1)')
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint(default: None)')
parser.add_argument('--test_every', type=int, default=10, metavar='N', help='test after every epochs')
parser.add_argument('--num_worker', type=int, default=1, metavar='N', help='the number of workers')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate(default: 0.001)')
parser.add_argument('--z_dim', type=int, default=20, metavar='N', help='the dim of latent variable z(default: 20)')
parser.add_argument('--input_dim', type=int, default=1 * 52, metavar='N', help='input dim(default: 28*28 for MNIST)')
parser.add_argument('--input_channel', type=int, default=1, metavar='N', help='input channel(default: 1 for MNIST)')
args = parser.parse_args()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


def dataloader(batch_size, num_workers):
    # adata = sc.read('data/cortex2.h5ad')
    cell_data = pd.read_csv('data/cortex3/obsm.csv')
    label_data = pd.read_csv('data/cortex3/obs.csv')
    label_data = label_data['label']
    cell_data = np.array(cell_data)[0:3000, :50]
    label_data = np.array(label_data)
    label = np.array([])
    for i in range(0, 3000):
        if label_data[i] == 'interneurons':
            label_data[i] = 0
        elif label_data[i] == 'pyramidal SS':
            label_data[i] = 1
        elif label_data[i] == 'pyramidal CA1':
            label_data[i] = 2
        elif label_data[i] == 'oligodendrocytes':
            label_data[i] = 3
        elif label_data[i] == 'microglia':
            label_data[i] = 4
        elif label_data[i] == 'endothelial-mural':
            label_data[i] = 5
        elif label_data[i] == 'astrocytes_ependymal':
            label_data[i] = 6
        label = np.append(label, label_data[i])
    data = torch.tensor(cell_data).float()
    label = torch.tensor(label).float()
    train = TensorDataset(data, label)
    # batch_size设置每一批数据的大小，shuffle设置是否打乱数据顺序，结果表明，该函数会先打乱数据再按batch_size取数据
    train = DataLoader(train, batch_size=batch_size, shuffle=False)
    test = DataLoader(train, batch_size=1, shuffle=False)
    classes = [0, 1, 2, 3, 4, 5, 6]
    # classes =('interneurons', 'pyramidal SS', 'pyramidal CA1', 'oligodendrocytes', 'microglia',
    #           'endothelial-mural', 'astrocytes_ependymal')
    return train, test, classes


def loss_function(x_hat, x, mu, log_var):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """
    # 1. the reconstruction loss.
    # Use likelihood as reconstruction loss
    # x = torch.sigmoid(x)
    # x_hat = torch.where(x_hat.gt(1), torch.full_like(x_hat, 1), x_hat)
    # x_hat = torch.sigmoid(x_hat)
    # BCE = F.binary_cross_entropy(x_hat, x, reduction='mean')
    # print(x.shape, x_hat.squeeze().shape)
    loss = torch.nn.MSELoss(reduction='mean')
    # loss = loss_criterion(x_hat, x)
    # loss = torch.nn.NLLLoss2d(reduction='sum')
    # loss = torch.nn.L1Loss(reduction='sum')
    # x_hat = x_hat.view(args.batch_size, 1,  -1)
    # x = x.view(args.batch_size, -1)
    # print(x.shape,x_hat.shape)
    ll = loss(x_hat, x)
    # loss_criterion = torch.nn.NLLLoss(weight=None, ignore_index=-100, reduction='sum')
    # likelihood = 0
    # print(x_hat.squeeze()[0, :].dim)
    # for i in range(x_hat.shape[0]):
    #     likelihood = likelihood + loss_criterion(x_hat.squeeze()[i, :], x.long()[i, :])
    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    # KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

    # 3. total loss
    # loss = likelihood
    # return loss, likelihood, KLD
    return ll
# def save_checkpoint(state, is_best, outdir):
#     """
#     每训练一定的epochs后， 判断损失函数是否是目前最优的，并保存模型的参数
#     :param state: 需要保存的参数，数据类型为dict
#     :param is_best: 说明是否为目前最优的
#     :param outdir: 保存文件夹
#     :return:
#     """
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#
#     checkpoint_file = os.path.join(outdir, 'checkpoint.pth')  # join函数创建子文件夹，也就是把第二个参数对应的文件保存在'outdir'里
#     best_file = os.path.join(outdir, 'model_best.pth')
#     torch.save(state, checkpoint_file)  # 把state保存在checkpoint_file文件夹中
#     if is_best:
#         shutil.copyfile(checkpoint_file, best_file)

def test(model, test):
    test_avg_loss = 0.0
    with torch.no_grad():  # 这一部分不计算梯度，也就是不放入计算图中去
        '''测试测试集中的数据'''
        # 计算所有batch的损失函数的和
        latent_z = np.array([])

        for test_batch_index, (test_x, _) in enumerate(test):
            # test_x = test_x.to(device)
            # 前向传播
            test_x_hat, test_mu, test_log_var, z = model(test_x)
            latent_z = np.append(latent_z, z)
            # 损害函数值
            test_loss, test_BCE, test_KLD = loss_function(test_x_hat, test_x, test_mu, test_log_var)
            test_avg_loss += test_loss

        sc.tl.umap(latent_z)
        sc.pl.umap(latent_z)
        # 对和求平均，得到每一张图片的平均损失
        test_avg_loss /= len(test.dataset)

        # '''测试随机生成的隐变量'''
        # 随机从隐变量的分布中取隐变量
        # z = torch.randn(args.batch_size, args.z_dim).to(device)  # 每一行是一个隐变量，总共有batch_size行
        # 对隐变量重构
        random_res = model.decode(z)
        # 保存重构结果
        # save_image(random_res, './%s/random_sampled-%d.png' % (args.result_dir, epoch + 1))

        # '''保存目前训练好的模型'''
        # # 保存模型
        # is_best = test_avg_loss < best_test_loss
        # best_test_loss = min(test_avg_loss, best_test_loss)
        # save_checkpoint({
        #     'epoch': epoch,  # 迭代次数
        #     'best_test_loss': best_test_loss,  # 目前最佳的损失函数值
        #     'state_dict': model.state_dict(),  # 当前训练过的模型的参数
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, args.save_dir)

        # return best_test_loss


def main():
    # Step 1: 载入数据
    # mnist_test, mnist_train, classes = dataloader(args.batch_size, args.num_worker)
    data_train, data_test, classes = dataloader(args.batch_size, args.num_worker)
    # 查看每一个batch的规模
    x, label = iter(data_train).__next__()  # 取出第一批(batch)训练所用的数据集
    print(' cell : ', x.shape)  # cell:torch.Size([batch_size, 50])，每次迭代获取batch_size cells，every cell has 50 components

    # Step 2: 准备工作 : 搭建计算流程
    model = VAE(batch_size=args.batch_size, z_dim=args.z_dim).to(device)  # 生成AE模型，并转移到GPU上去

    print('The structure of our model is shown below: \n')
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 生成优化器，需要优化的是model的参数，学习率为0.001

    # Step 3: optionally resume(恢复) from a checkpoint
    start_epoch = 0
    best_test_loss = np.finfo('f').max
    if args.resume:
        if os.path.isfile(args.resume):
            # 载入已经训练过的模型参数与结果
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Step 4: 开始迭代
    loss_epoch = []
    for epoch in range(start_epoch, args.epochs):
        # 训练模型
        # 每一代都要遍历所有的批次
        loss_batch = []
        for batch_index, (x, _) in enumerate(data_train):
            # x : [b, 50], remember to deploy the input on GPU
            # 前向传播
            x_hat, mu, log_var, latent_z = model(x)  # 模型的输出，在这里会自动调用model中的forward函数
            # print(batch_index, x_hat.shape, latent_z.shape)
            loss = loss_function(x_hat, x, mu, log_var)  # 计算损失值，即目标函数
            loss_batch.append(loss.item())  # loss是Tensor类型
            # 后向传播
            optimizer.zero_grad()  # 梯度清零，否则上一步的梯度仍会存在
            loss.backward()  # 后向传播计算梯度，这些梯度会保存在model.parameters里面
            optimizer.step()  # 更新梯度，这一步与上一步主要是根据model.parameters联系起来了

            # print statistics every 100 batch
            if (batch_index + 1) % 10 == 0:
                print('Epoch [{}/{}], Batch [{}/{}] : loss = {:.4f}'
                      .format(epoch + 1, args.epochs, batch_index + 1, 3000 // args.batch_size,
                              loss.item() / args.batch_size))

            # if batch_index == 0:
            #     # visualize reconstructed result at the beginning of each epoch
            #     x_concat = torch.cat([x.view(-1, 1, 28, 28), x_hat.view(-1, 1, 28, 28)], dim=3)
            #     save_image(x_concat, './%s/reconstructed-%d.png' % (args.result_dir, epoch + 1))

        # 把这一个epoch的每一个样本的平均损失存起来
        loss_epoch.append(np.sum(loss_batch) / 3000)  # len(mnist_train.dataset)为样本个数

    with torch.no_grad():  # 这一部分不计算梯度，也就是不放入计算图中去
        latent_z = np.array([])
        label_z = np.array([])
        for test_batch_index, (test_x, label) in enumerate(data_train):
            test_x_hat, test_mu, test_log_var, z = model(test_x)
            latent_z = np.append(latent_z, z)
            label_z = np.append(label_z, label)
        latent_z = torch.tensor(latent_z).view(-1, 20)
        reducer = UMAP(n_neighbors=15, n_components=2, metric='euclidean', learning_rate=1.0)
        latent_z = reducer.fit_transform(latent_z)
        print('Shape of X_trans: ', latent_z.shape)
        # torch.save(latent_z, "data/latent_z.pt")
        # torch.save(label_z, "data/labelsd_z.pt")
        ARI = kmeans.kmeanswithARI(latent_z, label_z)
        print('ARI score: ', ARI)
    return loss_epoch


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    loss_epoch = main()
    # 绘制迭代结果
    print(loss_epoch)
    plt.plot(loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
