from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import negative_binomial
import numpy as np
class VAE(nn.Module):

    def __init__(self, batch_size, input_dim=50, h_dim=200, z_dim=20):
        # 调用父类方法初始化模块的state
        super(VAE, self).__init__()
        self.batchsize = batch_size
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # 编码器 ： [b, input_dim] => [b, z_dim]
        input_dim = torch.tensor(input_dim, dtype=torch.int)
        h_dim = torch.tensor(h_dim, dtype=torch.int)
        self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        x = x.view(self.batchsize, -1)  # 一行代表一个样本
        x_hat, mu, log_var, sampled_z = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        for i in range(self.batchsize):
            # encoder
            mu_i, log_var_i = self.encode(x[i, :])
            mu = torch.cat((mu, mu_i), 0)
            log_var = torch.cat((log_var, log_var_i), 0)
            # reparameterization trick
            sampled_z_i = self.reparameterization(mu_i, log_var_i)
            sampled_z = torch.cat((sampled_z, sampled_z_i), 0)
            # decoder
            x_hat = torch.cat((x_hat, self.decode(sampled_z_i)), 0)
        # reshape
        sampled_z = sampled_z.view(batch_size, -1)
        x_hat = x_hat.view(batch_size, -1)  # 不知道多少，自己算吧
        return x_hat, mu, log_var, sampled_z

    def encode(self, x):
        """
        encoding part
        :param x: input cell
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        dec_theta = torch.exp(log_var)
        logits = (mu + 1e-6).log() - (dec_theta + 1e-6).log()
        logits = torch.where(torch.isnan(logits), torch.full_like(logits, 0), logits)
        NB = negative_binomial.NegativeBinomial(total_count=dec_theta, logits=torch.tensor(logits))
        z = NB.sample()
        return z
        # sigma = torch.exp(log_var * 0.5)
        # eps = torch.randn_like(sigma)
        # return mu + sigma * eps  # 这里的“*”是点乘的意思

    def decode(self, z):
        """
        Given a sampled z, decode it back to cell
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.relu(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU 就用relu
        return x_hat
