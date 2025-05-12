import torch
import torch.nn as nn
import torchvision.ops as ops
from e2cnn import gspaces, nn as enn
from torch_geometric.nn import GATConv


class SceneGraphEquiModel(nn.Module):
    def __init__(
        self,
        num_obj_classes,
        num_rel_classes,
        roi_size = int(7),
        gat_hidden = int(256),
        N = int(8), # number of rotations to be equivariant to
        spatial_scale = 1.0 / 4.0,
        sampling_ratio = int(2),
        aligned: bool = True
    ):
        super().__init__()

        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

        self.r2_act = gspaces.Rot2dOnR2(N=N)
        in_type  = enn.FieldType(self.r2_act,
                                 [self.r2_act.trivial_repr] * 3)
        hid_type = enn.FieldType(self.r2_act,
                                 [self.r2_act.regular_repr] * 16)

        self.conv1 = enn.R2Conv(
            in_type, hid_type,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1    = enn.InnerBatchNorm(hid_type)
        self.act1   = enn.ReLU(hid_type, inplace=True)

        self.pool1  = enn.PointwiseAvgPoolAntialiased(hid_type,
                                                      sigma=0.66,
                                                      stride=2)
        hid2_type = enn.FieldType(self.r2_act,
                                  [self.r2_act.regular_repr] * 32)
        self.conv2 = enn.R2Conv(hid_type, hid2_type,
                                kernel_size=3, padding=1, bias=False)
        self.bn2    = enn.InnerBatchNorm(hid2_type)
        self.act2   = enn.ReLU(hid2_type, inplace=True)
        self.pool2  = enn.PointwiseAvgPoolAntialiased(hid2_type,
                                                      sigma=0.66,
                                                      stride=2)
        out_type  = enn.FieldType(self.r2_act,
                                 [self.r2_act.trivial_repr] * 64)
        self.conv3 = enn.R2Conv(hid2_type, out_type,
                                kernel_size=1, bias=False)
        self.bn3    = enn.InnerBatchNorm(out_type)
        self.act3   = enn.ReLU(out_type, inplace=True)

        self.roi_align = ops.RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=spatial_scale,   
            sampling_ratio=sampling_ratio,

        )

        feat_dim = 64 * roi_size * roi_size
        self.box_mlp = nn.Sequential(
            nn.Linear(feat_dim, gat_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gat_hidden, gat_hidden),
            nn.ReLU(inplace=True),
        )

        self.gat1 = GATConv(
            in_channels=gat_hidden,
            out_channels=gat_hidden,
            heads=4,
            concat=True,
            dropout=0.2
        )
        self.gat2 = GATConv(
            in_channels=gat_hidden * 4,
            out_channels=gat_hidden,
            heads=1,
            concat=False,
            dropout=0.2
        )

        self.obj_head = nn.Linear(gat_hidden, num_obj_classes)
        self.rel_head = nn.Linear(2 * gat_hidden, num_rel_classes)

    def forward(self, images, boxes, edge_index):
        """
        images:    Tensor[B,3,H,W]
        boxes:     List[Tensor(#boxes_i,4)] of absolute coords
        edge_index: LongTensor[2,E] of global subj->obj indices
        """
        x = enn.GeometricTensor(images, 
                                enn.FieldType(self.r2_act,
                                              [self.r2_act.trivial_repr]*3))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        featmap = x.tensor            

        # RoI-Align & box MLP
        roi_feats = self.roi_align(
            featmap,
            boxes
            )  
        N = roi_feats.size(0)
        r = roi_feats.view(N, -1)                    
        h = self.box_mlp(r)                            

        # GAT + MLP
        h = self.gat1(h, edge_index)
        h = self.gat2(h, edge_index)
        obj_logits = self.obj_head(h)                  
        subj, predi = edge_index           
        edge_feat = torch.cat([h[subj], h[predi]], dim=1)
        rel_logits = self.rel_head(edge_feat)           

        return obj_logits, rel_logits