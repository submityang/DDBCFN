import torch
import torchvision.ops
from torch import nn
import copy


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding='SAME',
                 bias=False,
                 sps=None):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        # self.sps = copy.deepcopy(sps)

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class SPGDeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding='SAME',
                 segment=None,
                 bias=False):
        super(SPGDeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.segment = segment.tolist()

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def HDC(self, offset, centre_xlst, centre_ylst, true, weight):
        try:
            for centre_num, centre_x, centre_y in zip(true, centre_xlst, centre_ylst):
                lst = [self.segment[centre_x-1][centre_y-1], self.segment[centre_x-1][centre_y],
                       self.segment[centre_x-1][centre_y+1], self.segment[centre_x][centre_y-1],
                       self.segment[centre_x][centre_y], self.segment[centre_x][centre_y+1],
                       self.segment[centre_x+1][centre_y-1], self.segment[centre_x+1][centre_y],
                       self.segment[centre_x+1][centre_y+1]]
                for i in range(0, len(lst)):
                    if lst[i] != centre_num:
                        a, b = divmod(int(i), 3)
                        #weight[:][:][a][b] = 0
                        weight[:,:,a,b]=0
                #
                #
                # lst = torch.tensor(lst)
                # centre = torch.tensor(centre)
                # for feature in offset:
                #     mask = feature + lst
        except:
            return weight
        return weight

    def forward(self, source_x):
        h, w = source_x.shape[2:]
        max_offset = max(h, w) / 4.
        x = copy.deepcopy(source_x[:, 0:-1, :, :])
        loc_source = copy.deepcopy(source_x[:, -1, :, :])
        true = []
        centre_xlst = []
        centre_ylst = []
        for loc_ in loc_source:
            loc_ = torch.reshape(loc_, (25, 25))
            centre_loc = loc_[12][12]
            try:
                centre_x, centre_y = divmod(int(centre_loc), 1000)
                centre_num = self.segment[centre_x][centre_y]
            except:
                print(centre_x, centre_y)
            centre_xlst.append(centre_x)
            centre_ylst.append(centre_y)
            true.append(centre_num)
            # centre_x, centre_y = divmod(int(centre_loc), 1000)
            # centre_num = self.segment[centre_x-25][centre_y-25]
            # true.append(centre_num)

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        try:
            self.regular_conv.weight.data = self.HDC(offset, centre_xlst, centre_ylst, true, self.regular_conv.weight.data)
        finally:
            modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # print(self.regular_conv.weight)
        centre_xlst=torch.Tensor(centre_xlst)
        centre_ylst=torch.Tensor(centre_ylst)
        centre=(centre_xlst,centre_ylst)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )

        # for displacement, loc in zip(offset, loc_source):
        #     loc_x, loc_y = divmod(int(loc[12][12]), 1000)
        #     cls_centre = self.segment[loc_x-25][loc_y-25]
        #     for i in range(0, (len(displacement) // 2)):
        #         dis = 100 * displacement[2 * i] + displacement[2 * i + 1]
        return x

    # def getclass(self, loc_x, loc_y):
