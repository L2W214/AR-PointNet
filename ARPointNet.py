import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from attention_mechanism import ChannelStatisticalAttention, SpatialStatisticalAttention


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=(1,))

        self.fc1 = nn.Linear(1024, 512)  # 用于设置网络中的全连接层的
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        pass

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # reshape成1024列，但是不确定几行

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        i_den = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1,
                                                    0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batch_size, 1)

        if x.is_cuda:
            i_den = i_den.cuda()
            pass

        x = x + i_den
        x = x.view(-1, 3, 3)  # 输出为Batch * 3 * 3的张量

        return x
    pass


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=k, out_channels=64, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=(1,))

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k
        pass

    def forward(self, x):
        batch_size = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        i_den = Variable(torch.from_numpy(np.eye(self.k).flatten().
                                          astype(np.float32))).view(1, self.k*self.k).repeat(batch_size, 1)

        if x.is_cuda:
            i_den = i_den.cuda()
            pass

        x = x + i_den
        x = x.view(-1, self.k, self.k)

        return x
    pass


class PointNetFeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetFeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(1,))
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=(1,))
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=(1,))

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.f_stn = STNkd()

        self.att1 = ChannelStatisticalAttention(64, 8)
        self.att2 = SpatialStatisticalAttention(64)

        # if self.feature_transform:
        #     self.f_stn = STNkd()
        #     pass
        pass

    def forward(self, x):
        """
        生成全局特征
        """
        n_pts = x.size()[2]
        trans_ = self.stn(x)

        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_)  # 计算两个tensor的矩阵乘法
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        if self.feature_transform:
            trans_feat = self.f_stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.reshape(2, 1)
            pass
        else:
            trans_feat = None
            pass

        point_feat = x
        x = self.att1(x)
        x = self.att2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        x = torch.max(x, 2, keepdim=True)[0]  # batch * 1024 * 1
        x = x.view(-1, 1024)

        # if self.global_feat:
        #     return x, trans_, trans_feat
        # else:
        #     x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        #     return torch.cat([x, point_feat], 1), trans_, trans_feat

        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)

        return torch.cat([x, point_feat], 1), trans_, trans_feat, point_feat
    pass


class SlicePool(nn.Module):
    def __init__(self, r, dim):
        super(SlicePool, self).__init__()

        self.r = r
        self.dim = dim

        # self.att = SCIAttention(64, 8)
        pass

    def forward(self, x, x_64, batch_, points):
        point_max = x.max(2)[0]
        point_min = x.min(2)[0]

        if self.dim == 'x':
            class_n = ((point_max[:, 0] - point_min[:, 0]) / self.r).unsqueeze(1)
            class_n = torch.ceil(class_n)
            class_n_max = class_n.max()
            point = x[:, 0, :]
            k = torch.floor((point - point_min[:, 0].unsqueeze(1)) / self.r)
            k = k.view(batch_, 1, points)
            pass
        elif self.dim == 'y':
            class_n = ((point_max[:, 1] - point_min[:, 1]) / self.r).unsqueeze(1)
            class_n = torch.ceil(class_n)
            class_n_max = class_n.max()
            point = x[:, 1, :]
            k = torch.floor((point - point_min[:, 1].unsqueeze(1)) / self.r)
            k = k.view(batch_, 1, points)
            pass
        else:
            class_n = ((point_max[:, 2] - point_min[:, 2]) / self.r).unsqueeze(1)
            class_n = torch.ceil(class_n)
            class_n_max = class_n.max()
            point = x[:, 2, :]
            k = torch.floor((point - point_min[:, 2].unsqueeze(1)) / self.r)
            k = k.view(batch_, 1, points)
            pass

        point_idx = torch.cat([x_64, k], 1)

        point_idx = point_idx.cpu()
        point_idx = point_idx.detach().numpy()
        max_feature, slice_num = [], []
        for i_ in range(int(class_n_max)):
            point_col_row = np.where(point_idx[:, -1, :] == i_)
            point_ = point_idx[point_col_row[0], 0:64, point_col_row[1]]
            point_ = torch.tensor(point_)
            point_ = point_.unsqueeze(2)
            if point_.size()[0] != 0:
                slice_num.append(point_.size()[0])
                point_ = point_.view(64, -1)
                feature = torch.max(point_, 1, keepdim=True)[0]  # 64, 1
                feature = feature.detach().numpy()
                max_feature.append(feature)
                pass
            pass

        max_feature = torch.tensor(max_feature)

        return max_feature, slice_num
    pass


def slice_un_pool(slice_num, rnn_feature, batch_num, points_num):
    rnn_feature = rnn_feature.cpu()
    temp = []
    for i, num in enumerate(slice_num):
        temp_tensor = rnn_feature[i, :, :]
        temp_tensor = temp_tensor.repeat(num, 1)

        temp.append(temp_tensor)
        pass

    un_pool = temp[0]
    for i in range(1, len(temp)):
        un_pool = torch.cat((un_pool, temp[i]), 0)
        pass

    size_0 = batch_num * points_num
    if un_pool.size()[0] != size_0:
        diff = size_0 - un_pool.size()[0]
        zero_temp = torch.zeros(diff, 512)
        un_pool = torch.cat([un_pool, zero_temp], 0)
        pass

    un_pool = un_pool.view(batch_num, points_num, 512)

    return un_pool


class RecurrentSlice(nn.Module):
    def __init__(self, r):
        super(RecurrentSlice, self).__init__()

        self.sp_x = SlicePool(r, 'x')
        self.sp_y = SlicePool(r, 'y')
        self.sp_z = SlicePool(r, 'z')

        self.rnn_x1 = nn.LSTM(64, 256, 1, bidirectional=True)
        self.rnn_x2 = nn.LSTM(512, 128, 1, bidirectional=True)
        self.rnn_x3 = nn.LSTM(256, 64, 1, bidirectional=True)
        self.rnn_x4 = nn.LSTM(128, 64, 1, bidirectional=True)
        self.rnn_x5 = nn.LSTM(128, 128, 1, bidirectional=True)
        self.rnn_x6 = nn.LSTM(256, 256, 1, bidirectional=True)

        self.rnn_y1 = nn.LSTM(64, 256, 1, bidirectional=True)
        self.rnn_y2 = nn.LSTM(512, 128, 1, bidirectional=True)
        self.rnn_y3 = nn.LSTM(256, 64, 1, bidirectional=True)
        self.rnn_y4 = nn.LSTM(128, 64, 1, bidirectional=True)
        self.rnn_y5 = nn.LSTM(128, 128, 1, bidirectional=True)
        self.rnn_y6 = nn.LSTM(256, 256, 1, bidirectional=True)

        self.rnn_z1 = nn.LSTM(64, 256, 1, bidirectional=True)
        self.rnn_z2 = nn.LSTM(512, 128, 1, bidirectional=True)
        self.rnn_z3 = nn.LSTM(256, 64, 1, bidirectional=True)
        self.rnn_z4 = nn.LSTM(128, 64, 1, bidirectional=True)
        self.rnn_z5 = nn.LSTM(128, 128, 1, bidirectional=True)
        self.rnn_z6 = nn.LSTM(256, 256, 1, bidirectional=True)
        pass

    def forward(self, x, x_64):
        batch_, dimension_, point_num = x.size()

        max_feature_x, slice_num_x = self.sp_x(x, x_64, batch_, point_num)  # numSlices, 64, 1
        max_feature_y, slice_num_y = self.sp_y(x, x_64, batch_, point_num)  # numSlices, 64, 1
        max_feature_z, slice_num_z = self.sp_z(x, x_64, batch_, point_num)  # numSlices, 64, 1

        max_feature_x = max_feature_x.transpose(2, 1)  # numSlices, 1, 64
        max_feature_y = max_feature_y.transpose(2, 1)  # numSlices, 1, 64
        max_feature_z = max_feature_z.transpose(2, 1)  # numSlices, 1, 64
        max_feature_x, max_feature_y, max_feature_z = max_feature_x.cuda(), max_feature_y.cuda(), max_feature_z.cuda()

        x_rnn_1, _ = self.rnn_x1(max_feature_x)
        x_rnn_2, _ = self.rnn_x2(x_rnn_1)
        x_rnn_3, _ = self.rnn_x3(x_rnn_2)
        x_rnn_4, _ = self.rnn_x4(x_rnn_3)
        x_rnn_5, _ = self.rnn_x5(x_rnn_4)
        x_rnn_6, _ = self.rnn_x6(x_rnn_5)  # numSlices, 1, 512

        y_rnn_1, _ = self.rnn_y1(max_feature_y)
        y_rnn_2, _ = self.rnn_y2(y_rnn_1)
        y_rnn_3, _ = self.rnn_y3(y_rnn_2)
        y_rnn_4, _ = self.rnn_y4(y_rnn_3)
        y_rnn_5, _ = self.rnn_y5(y_rnn_4)
        y_rnn_6, _ = self.rnn_y6(y_rnn_5)  # numSlices, 1, 512

        z_rnn_1, _ = self.rnn_z1(max_feature_z)
        z_rnn_2, _ = self.rnn_z2(z_rnn_1)
        z_rnn_3, _ = self.rnn_z3(z_rnn_2)
        z_rnn_4, _ = self.rnn_z4(z_rnn_3)
        z_rnn_5, _ = self.rnn_z5(z_rnn_4)
        z_rnn_6, _ = self.rnn_z6(z_rnn_5)  # numSlices, 1, 512

        x_rnn_6 = slice_un_pool(slice_num_x, x_rnn_6, batch_, point_num)  # batch_, point_num, 512
        y_rnn_6 = slice_un_pool(slice_num_y, y_rnn_6, batch_, point_num)
        z_rnn_6 = slice_un_pool(slice_num_z, z_rnn_6, batch_, point_num)

        rnn_all = x_rnn_6 + y_rnn_6 + z_rnn_6
        rnn_all = rnn_all.cuda()
        rnn_all = rnn_all.transpose(2, 1)  # batch_, 512, point_num

        return rnn_all
    pass


class ARPointNet(nn.Module):
    def __init__(self, k, r, feature_transform=False):
        super(ARPointNet, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetFeat(global_feat=True, feature_transform=feature_transform)

        self.rs = RecurrentSlice(r)

        self.conv1_ar = nn.Conv1d(in_channels=2112, out_channels=1024, kernel_size=(1,))
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=(1,))
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(1,))
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=self.k, kernel_size=(1,))

        self.bn1_ar = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv_4_ar = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=(1,))
        self.bn_3_ar = nn.BatchNorm1d(1024)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.conv_1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=(1,))
        self.conv_2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=(1,))
        self.conv_3 = nn.Conv1d(in_channels=256, out_channels=self.k, kernel_size=(1,))

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)

        self.drop = nn.Dropout(p=0.3)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pass

    def forward(self, x):
        point = x
        batch_size = x.size()[0]
        n_pts = x.size()[2]
        x, trans_, trans_feat, point_feat = self.feat(x)

        rnn_all = self.rs(point, point_feat)  # batch_, 512, point_num

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        x_slice = F.relu(self.bn_1(self.conv_1(rnn_all)))
        x_slice = F.relu(self.bn_2(self.conv_2(x_slice)))
        x_slice = self.drop(x_slice)
        x_slice = self.conv_3(x_slice)

        x_slice = x_slice.transpose(2, 1).contiguous()  # 调用contiguous()之后，会真正改变Tensor的内容，按照变换之后的顺序存放数据。
        x_slice = F.log_softmax(x_slice.view(-1, self.k), dim=-1)  # dim=-1是最后一个维度
        x_slice = x_slice.view(batch_size, n_pts, self.k)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        rnn_all = F.relu(self.bn_3_ar(self.conv_4_ar(rnn_all)))
        x = torch.cat([x, rnn_all], 1)  # batch * 2112 * n_pts

        x = F.relu(self.bn1_ar(self.conv1_ar(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)

        x = x.transpose(2, 1).contiguous()  # 调用contiguous()之后，会真正改变Tensor的内容，按照变换之后的顺序存放数据。
        x = F.log_softmax(x.view(-1, self.k), dim=-1)  # dim=-1是最后一个维度
        x = x.view(batch_size, n_pts, self.k)

        return x, trans_, trans_feat, x_slice
    pass


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    # batch_size = trans.size()[0]
    e = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        e = e.cuda()
        pass

    temp = torch.bmm(trans, trans.transpose(2, 1)) - e
    # temp = torch.FloatTensor(temp)
    loss = torch.mean(torch.norm(temp, dim=[1, 2]))

    return loss
