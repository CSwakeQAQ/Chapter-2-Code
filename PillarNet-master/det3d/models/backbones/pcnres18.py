import torch
from torch import nn

# 导入注册器和一些通用构建函数和模块
from ..registry import BACKBONES
from ..utils import build_norm_layer
from .base import spconv, SparseConv2d, Sparse2DBasicBlock, Sparse2DBasicBlockV, post_act_block_dense

# 注册骨干网络模块
@BACKBONES.register_module
class SpMiddlePillarEncoder(nn.Module):
    """
    稀疏卷积中间编码器，用于点云特征提取。
    架构包含四层稀疏卷积模块，逐步提取空间特征，最终转换为密集表示。
    """
    def __init__(self, in_planes=32, name="SpMiddlePillarEncoder", **kwargs):
        super(SpMiddlePillarEncoder, self).__init__()
        self.name = name

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)  # 设置标准化参数

        # 第一层：两个残差块
        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        # 第二层：卷积降采样 + 残差块
        self.conv2 = spconv.SparseSequential(
            SparseConv2d(32, 64, 3, 2, padding=1, bias=False),  # 空间尺寸缩小一半
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        # 第三层：进一步降采样并加深通道
        self.conv3 = spconv.SparseSequential(
            SparseConv2d(64, 128, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        # 第四层：最后一次稀疏降采样
        self.conv4 = spconv.SparseSequential(
            SparseConv2d(128, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
        )

        # 定义每层输出的通道数和步长（相对于原始输入）
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
        }
        self.backbone_strides = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
        }

    def forward(self, sp_tensor):
        # 网络前向传播流程
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()  # 将稀疏表示转换为密集表示，用于后续处理

        return x_conv4
