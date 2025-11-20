"""
BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment, CVPR 2022
"""

import torch
import torch.nn as nn

try:
    from mmcv.cnn import constant_init
    from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
except:
    from loguru import logger

    logger.warning("mmcv is not available, using a fallback implementation")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Fallback: constant_init function
    def constant_init(module, val=0, bias=0):
        """Initialize module parameters with constant values."""
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _pair(v):
        if isinstance(v, tuple):
            return v
        return (v, v)

    @torch.no_grad()
    def _compute_output_shape(H_in, W_in, kH, kW, stride, padding, dilation):
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation
        H_out = (H_in + 2 * ph - (dh * (kH - 1) + 1)) // sh + 1
        W_out = (W_in + 2 * pw - (dw * (kW - 1) + 1)) // sw + 1
        return H_out, W_out

    def _modulated_deform_conv2d_core(
        x,
        offset,
        mask,
        weight,
        bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
    ):
        """
        纯 PyTorch 版 Modulated Deformable Conv2d，支持 deform_groups
        x:      (N, C_in, H_in, W_in)
        offset: (N, 2*kH*kW*deform_groups, H_out, W_out)
        mask:   (N,   kH*kW*deform_groups, H_out, W_out)
        weight: (C_out, C_in/groups, kH, kW)
        bias:   (C_out,) or None
        """
        if groups != 1:
            raise NotImplementedError("This fallback only supports groups=1 for now.")

        device = x.device
        dtype = x.dtype

        N, C_in, H_in, W_in = x.shape
        C_out, C_in_w, kH, kW = weight.shape
        assert C_in_w == C_in, "groups=1 only, C_in in weight must equal input channels"

        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation

        # conv 输出空间大小
        H_out, W_out = _compute_output_shape(H_in, W_in, kH, kW, stride, padding, dilation)

        # 检查 offset / mask 形状
        K = kH * kW
        assert offset.shape[1] == 2 * K * deform_groups
        assert offset.shape[2] == H_out and offset.shape[3] == W_out
        assert mask.shape[1] == K * deform_groups
        assert mask.shape[2] == H_out and mask.shape[3] == W_out

        # 每个 deform_group 负责的输入通道数
        C_per_deform_group = C_in // deform_groups

        # 输出
        out = x.new_zeros(N, C_out, H_out, W_out)

        # 输出位置网格 (h_out, w_out)
        yy, xx = torch.meshgrid(
            torch.arange(H_out, device=device, dtype=dtype),
            torch.arange(W_out, device=device, dtype=dtype),
            indexing="ij",
        )  # (H_out, W_out)

        # batch 索引，用于高级索引
        n_idx = torch.arange(N, device=device).view(N, 1, 1).expand(N, H_out, W_out)

        # 遍历每个 deform_group
        for g in range(deform_groups):
            # 该 group 负责的输入通道范围
            c_start = g * C_per_deform_group
            c_end = (g + 1) * C_per_deform_group
            x_g = x[:, c_start:c_end, :, :]  # (N, C_per_deform_group, H_in, W_in)

            for i in range(kH):
                for j in range(kW):
                    k = i * kW + j  # 第 k 个 kernel 位置

                    # 该 group 和 kernel 位置对应的 offset / mask 索引
                    offset_idx = g * K + k

                    # 常规 conv 的采样中心位置 p0 + p_n
                    base_y = yy * sh - ph + i * dh  # (H_out, W_out)
                    base_x = xx * sw - pw + j * dw  # (H_out, W_out)

                    # 取出该 group+kernel 位置对应的 offset 分量
                    off_y = offset[:, 2 * offset_idx + 0, :, :]  # (N, H_out, W_out)
                    off_x = offset[:, 2 * offset_idx + 1, :, :]  # (N, H_out, W_out)

                    # 最终采样坐标
                    pos_y = base_y.unsqueeze(0) + off_y  # (N, H_out, W_out)
                    pos_x = base_x.unsqueeze(0) + off_x  # (N, H_out, W_out)

                    # 双线性插值的 4 个邻居坐标（浮点）
                    y0 = torch.floor(pos_y)
                    x0 = torch.floor(pos_x)
                    y1 = y0 + 1
                    x1 = x0 + 1

                    # 是否在合法范围内（用于零填充）
                    inside = (pos_y >= 0) & (pos_y <= H_in - 1) & (pos_x >= 0) & (pos_x <= W_in - 1)

                    # clamp 之后再索引
                    y0c = y0.clamp(0, H_in - 1).long()
                    y1c = y1.clamp(0, H_in - 1).long()
                    x0c = x0.clamp(0, W_in - 1).long()
                    x1c = x1.clamp(0, W_in - 1).long()

                    # 双线性权重
                    wy0 = (y1 - pos_y).clamp(0, 1)
                    wy1 = (pos_y - y0).clamp(0, 1)
                    wx0 = (x1 - pos_x).clamp(0, 1)
                    wx1 = (pos_x - x0).clamp(0, 1)

                    wa = (wy0 * wx0)[:, None, :, :]  # (N,1,H_out,W_out)
                    wb = (wy0 * wx1)[:, None, :, :]
                    wc = (wy1 * wx0)[:, None, :, :]
                    wd = (wy1 * wx1)[:, None, :, :]

                    # 从 x_g 中取四个邻居值
                    Ia = x_g[n_idx, :, y0c, x0c].permute(
                        0, 3, 1, 2
                    )  # (N, C_per_deform_group, H_out, W_out)
                    Ib = x_g[n_idx, :, y0c, x1c].permute(0, 3, 1, 2)
                    Ic = x_g[n_idx, :, y1c, x0c].permute(0, 3, 1, 2)
                    Id = x_g[n_idx, :, y1c, x1c].permute(0, 3, 1, 2)

                    # 双线性插值
                    sampled = (
                        Ia * wa + Ib * wb + Ic * wc + Id * wd
                    )  # (N, C_per_deform_group, H_out, W_out)

                    # 越界位置置 0
                    sampled = sampled * inside[:, None, :, :]

                    # modulation mask（这一位置对应的 mask 标量）
                    mask_k = mask[:, offset_idx, :, :]  # (N, H_out, W_out)
                    sampled = sampled * mask_k[:, None, :, :]

                    # 该 kernel 位置和 group 对应的权重 (C_out, C_per_deform_group)
                    w_ij = weight[:, c_start:c_end, i, j]  # (C_out, C_per_deform_group)

                    # 按公式累加：out += w_ij @ sampled
                    out = out + torch.einsum("oc,nchw->nohw", w_ij, sampled)

        if bias is not None:
            out = out + bias.view(1, -1, 1, 1)

        return out

    def modulated_deform_conv2d(
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
    ):
        """
        函数版接口，对齐 mmcv.ops.modulated_deform_conv2d
        """
        return _modulated_deform_conv2d_core(
            input,
            offset,
            mask,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
        )

    class ModulatedDeformConv2d(nn.Module):
        """
        纯 PyTorch 版 ModulatedDeformConv2d，接口对齐 mmcv.ops.ModulatedDeformConv2d
        支持 deform_groups，但仅支持 groups=1

        forward(x, offset, mask) -> y
        """

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            deform_groups=1,
            bias=True,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = _pair(kernel_size)
            self.stride = _pair(stride)
            self.padding = _pair(padding)
            self.dilation = _pair(dilation)
            self.groups = groups
            self.deform_groups = deform_groups

            if groups != 1:
                raise NotImplementedError(
                    "This fallback ModulatedDeformConv2d currently only supports groups=1."
                )

            C_in_group = in_channels // groups
            self.weight = nn.Parameter(torch.empty(out_channels, C_in_group, *self.kernel_size))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_channels))
            else:
                self.register_parameter("bias", None)

            # 简单初始化，也可以在外面用 constant_init / kaiming_init 再调
            nn.init.kaiming_uniform_(self.weight, a=1.0)

        def forward(self, x, offset, mask):
            return _modulated_deform_conv2d_core(
                x,
                offset,
                mask,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                deform_groups=self.deform_groups,
            )


from src.demark_world.models.model.modules.flow_comp import flow_warp


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel):
        super(BidirectionalPropagation, self).__init__()
        modules = ["backward_", "forward_"]
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * channel, channel, 3, padding=1, deform_groups=16
            )

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)

    def forward(self, x, flows_backward, flows_forward):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """
        b, t, c, h, w = x.shape
        feats = {}
        feats["spatial"] = [x[:, i, :, :, :] for i in range(0, t)]

        for module_name in ["backward_", "forward_"]:
            feats[module_name] = []

            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            mapping_idx = list(range(0, len(feats["spatial"])))
            mapping_idx += mapping_idx[::-1]

            if "backward" in module_name:
                frame_idx = frame_idx[::-1]
                flows = flows_backward
            else:
                flows = flows_forward

            feat_prop = x.new_zeros(b, self.channel, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats["spatial"][mapping_idx[idx]]

                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]
                    cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                    # initialize second-order features
                    feat_n2 = torch.zeros_like(feat_prop)
                    flow_n2 = torch.zeros_like(flow_n1)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                        flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                        cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

                feat = (
                    [feat_current]
                    + [feats[k][idx] for k in feats if k not in ["spatial", module_name]]
                    + [feat_prop]
                )

                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)
                feats[module_name].append(feat_prop)

            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != "spatial"]
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + x
