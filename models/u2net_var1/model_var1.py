from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop() 
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class GateFusion(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(GateFusion, self).__init__()

        self.gate_1 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        cat_fea = torch.cat([x1, x2], dim=1) #128,32,32

        att_vec_1 = self.gate_1(cat_fea)
        att_vec_2 = self.gate_2(cat_fea)

        att_vec_cat = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)

        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2

        return x_fusion


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out_mid = max_out
        out = self.sigmoid(out_mid)
        return out


class global_module(nn.Module):
    def __init__(self, channels=64, r=4):
        super(global_module, self).__init__()
        out_channels = int(channels // r)
        # local_att

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xg = self.global_att(x)
        out = self.sig(xg)

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BAM(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(BAM, self).__init__()

        self.relu = nn.ReLU(True)

        self.global_att = global_module(channel)

        self.conv_layer = BasicConv2d(channel * 2, channel, 3, padding=1)

    def forward(self, x, x_boun_atten):
        out1 = self.conv_layer(torch.cat((x, x_boun_atten), dim=1))#64,32,32
        out2 = self.global_att(out1) #64,1,1
        out3 = out1.mul(out2) #64,32,32

        out = x + out3 #64,32,32

        return out


class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1, edge_ch = 64):
        super(U2Net,self).__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])

        # for CFA edge info
        act_fn = nn.ReLU(inplace=True)
        self.low_fusion = GateFusion(edge_ch)
        self.layer_edge0 = nn.Sequential(nn.Conv2d(edge_ch, edge_ch, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(edge_ch), act_fn)
        self.layer_edge1 = nn.Sequential(nn.Conv2d(edge_ch, edge_ch, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(edge_ch), act_fn)
        self.layer_edge2 = nn.Sequential(nn.Conv2d(edge_ch, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
                                         act_fn)
        self.layer_edge3 = nn.Sequential(nn.Conv2d(64, 5, kernel_size=1))

        
        self.atten_edge_0 = ChannelAttention(edge_ch)
        self.atten_edge_1 = ChannelAttention(edge_ch)
        self.atten_edge_2 = ChannelAttention(edge_ch)
        self.atten_edge_ori = ChannelAttention(edge_ch)

        # BAM
        self.cat_01 = BAM(edge_ch)
        self.cat_11 = BAM(edge_ch)
        self.cat_21 = BAM(edge_ch)
        self.cat_31 = BAM(edge_ch)

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # encode part
        encode_list = []
        side_list = []
        for c in cfg["encode"]:
            assert len(c) == 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4])) 

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        # decode part
        decode_list = []
        for c in cfg["decode"]:
            assert len(c) == 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        # collect encode outputs
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # get the edge info
        # for lite model, output chanel is 64, do not need to add conv layer
        en1 = encode_outputs[0] #64, 256, 256
        en1_out = F.max_pool2d(en1, kernel_size=8, stride=8, ceil_mode=True)# 64,32,32
        
        en3 = encode_outputs[2] #64, 64, 64
        en3_out = F.max_pool2d(en3, kernel_size=2, stride=2, ceil_mode=True)# 64,32,32

       

        low_x = self.low_fusion(en1_out, en3_out) #output: 64,32,32
        edge_out0 = self.layer_edge0(self.up_2(low_x))  # 64,64,64
        edge_out1 = self.layer_edge1(self.up_2(edge_out0))  # 64, 128, 128
        edge_out2 = self.layer_edge2(self.up_2(edge_out1))  # 64, 256, 256
        edge_out3 = self.layer_edge3(edge_out2)  # 5, 256, 256

        

        # chanel attention
        etten_edge_ori = self.atten_edge_ori(low_x)  # 64,1,1
        etten_edge_0 = self.atten_edge_0(edge_out0)  # 64,1,1
        etten_edge_1 = self.atten_edge_1(edge_out1)  # 64,1,1
        etten_edge_2 = self.atten_edge_2(edge_out2)  # 64,1,1




        # collect decode outputs
        x = encode_outputs.pop() #64,8,8
        decode_outputs = [x]

        x2 = encode_outputs.pop() #64,16,16
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False) # 64,16,16
        m0 = self.decode_modules[0] # RSU4F
        x = m0(torch.concat([x, x2], dim=1)) #64,16,16
        decode_outputs.insert(0, x) # 

        x2 = encode_outputs.pop() #64,32,32
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False) #64,32,32
        cat_out_01 = self.cat_01(x, low_x.mul(etten_edge_ori))  # BAM model 64,32,32
        x = cat_out_01  #64,32,32
        m1= self.decode_modules[1] #RSU
        x = m1(torch.concat([x, x2], dim=1)) #64,32,32
        decode_outputs.insert(0, x)

        x2 = encode_outputs.pop() #64,64,64
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False) #64,64,64
        cat_out_11 = self.cat_11(x, edge_out0.mul(etten_edge_0)) # BAM model 64,64,64
        x = cat_out_11 #64,64,64
        m2 = self.decode_modules[2] #RSU
        x = m2(torch.concat([x, x2], dim=1)) #64,64,64
        decode_outputs.insert(0, x)

        x2 = encode_outputs.pop() #64,128,128
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False) #64,128,128
        cat_out_21 = self.cat_21(x, edge_out1.mul(etten_edge_1)) #BAM model 64,128,128
        x = cat_out_21 #64,128,128
        m3 = self.decode_modules[3]
        x = m3(torch.concat([x, x2], dim=1)) #RSU
        decode_outputs.insert(0, x)

        x2 = encode_outputs.pop() #64,256,256
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False) #64,256,256
        cat_out_31 = self.cat_31(x, edge_out2.mul(etten_edge_2)) #BAM 64,256,256
        x = cat_out_31 #64,256,256
        m4 = self.decode_modules[4] #RSU
        x = m4(torch.concat([x, x2], dim=1)) #64,256,256
        decode_outputs.insert(0, x)



        # collect side outputs
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.concat(side_outputs, dim=1))

        return [x] + side_outputs + [edge_out3]






def u2net_lite(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1


    }

    return U2Net(cfg, out_ch)



def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 3, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


if __name__ == '__main__':
   
    u2net = u2net_lite()
    convert_onnx(u2net, "u2net_full.onnx")
