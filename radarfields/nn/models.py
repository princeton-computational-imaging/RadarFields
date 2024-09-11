import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np

from radarfields.nn.encoding import PositionalEncoding
from radarfields.nn.tcnn_utils import get_encoding_config, get_mlp_config

def mask(encoding, mask_coef):
    #NOTE: borrowed from NSF (Chugunov et al.): https://github.com/princeton-computational-imaging/NSF 

    if mask_coef is None: return encoding

    mask_coef = 0.4 + 0.6*mask_coef

    # Interpolate to size of encoding
    mask = torch.zeros_like(encoding[0:1])
    mask_ceil = int(np.ceil(mask_coef * encoding.shape[1]))
    mask[:,:mask_ceil] = 1.0
    return encoding * mask

def linear(in_params, out_params, relu=True, leaky=False, slope=0.01, bn=False):

    layer = [nn.Linear(in_features=in_params,out_features=out_params)]

    if bn: layer.append(nn.BatchNorm1d(out_params))

    if relu:
        if leaky: layer.append(nn.LeakyReLU(negative_slope=slope))
        else: layer.append(nn.ReLU())

    return nn.Sequential(*layer)

class MLP(nn.Module):
    def __init__(self, in_dim, num_layers, hidden_dim, out_dim, bn=False):
        super().__init__()

        self.model = nn.Sequential(linear(in_dim, hidden_dim, bn), 
                                   *[linear(hidden_dim, hidden_dim, bn) 
                                     for _ in range(num_layers-2)],
                                     linear(hidden_dim, out_dim, relu=False))
    
    def forward(self, x):
        return self.model(x)
 
class RadarField(nn.Module):
    def __init__(self, 
                 in_dim=3,
                 xyz_encoding="HashGrid",
                 num_layers=6,
                 hidden_dim=64,
                 xyz_feat_dim=20,
                 alpha_dim=1,
                 alpha_activation="sigmoid",
                 sigmoid_tightness=8.0,
                 rd_dim=1,
                 softplus_rd=True,
                 angle_dim=3,
                 angle_in_layer=5,
                 angle_encoding="SphericalHarmonics",
                 num_bands_xyz=10,
                 resolution=2048,
                 n_levels=16,
                 bound=1,
                 bn=False,
                 use_tcnn=True
                 ):
        super().__init__()
        self.alpha_activation = alpha_activation
        self.softplus_rd = softplus_rd
        self.hashgrid = xyz_encoding == "HashGrid"
        self.bn = bn
        self.tcnn = use_tcnn

        # Activations
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_tightness = sigmoid_tightness
        self.softplus = nn.Softplus()

        if self.tcnn: # Instantiating network w TCNN

            if angle_encoding == "HashGrid": raise ValueError("ERROR: angle_encoding cannot be HashGrid.")

            # TCNN configs
            self.encode_xyz_config = get_encoding_config(xyz_encoding, resolution, bound,
                                                         n_bands=num_bands_xyz, n_levels=n_levels)
            self.encode_angle_config = get_encoding_config(angle_encoding, None, None)
            self.xyz_net_config = get_mlp_config(hidden_dim, angle_in_layer - 1)
            self.alpha_net_config = get_mlp_config(hidden_dim, num_layers - angle_in_layer + 1)
            self.rd_net_config = get_mlp_config(hidden_dim, num_layers - angle_in_layer + 1)
            
            # Encodings & NNs
            self.encode_xyz = tcnn.Encoding(n_input_dims=in_dim, encoding_config=self.encode_xyz_config)
            self.encode_angle = tcnn.Encoding(n_input_dims=angle_dim, encoding_config=self.encode_angle_config)
            self.xyz_net = tcnn.Network(n_input_dims=self.encode_xyz.n_output_dims,
                                        n_output_dims=xyz_feat_dim,
                                        network_config=self.xyz_net_config)
            self.alpha_net = tcnn.Network(n_input_dims=xyz_feat_dim,
                                          n_output_dims=alpha_dim,
                                          network_config=self.alpha_net_config)
            self.rd_net = tcnn.Network(n_input_dims=self.encode_angle.n_output_dims + xyz_feat_dim,
                                        n_output_dims=rd_dim,
                                        network_config=self.rd_net_config)

            if bn: self.xyz_net = nn.Sequential(self.xyz_net, nn.BatchNorm1d(xyz_feat_dim))

            return

        # Encodings & NNs **without TCNN**
        self.encode_xyz = PositionalEncoding(in_dim, b=num_bands_xyz)
        self.encode_angle = PositionalEncoding(angle_dim)
        self.xyz_net = MLP(self.encode_xyz.o_dim,
                       angle_in_layer - 1,
                       hidden_dim,
                       xyz_feat_dim,
                       bn=bn)
        self.alpha_net = MLP(xyz_feat_dim,
                             num_layers - angle_in_layer + 1,
                             hidden_dim,
                             alpha_dim)
        self.rd_net = MLP(self.encode_angle.o_dim + xyz_feat_dim,
                           num_layers - angle_in_layer + 1,
                           hidden_dim,
                           rd_dim)
    
    def forward(self, xyz, angle, sin_epoch=None):  
        out = {}

        I = xyz.shape[-1]
        original_shape = tuple(list(xyz.shape)[:-1] + [-1])
        xyz = xyz.reshape((-1, I))
        angle = angle.reshape((-1, I))

        xyz_encoded = self.encode_xyz(xyz)
        angle_encoded = self.encode_angle(angle)

        # Coarse-to-fine mask on HashGrid encoding
        if self.hashgrid: xyz_encoded = mask(xyz_encoded, sin_epoch)

        xyz_features = self.xyz_net(xyz_encoded)
        alpha = self.alpha_net(xyz_features)
        rd = self.rd_net(torch.cat((angle_encoded, xyz_features), dim=-1))

        # Optional activations
        if self.alpha_activation == "softplus":
            alpha = self.softplus(alpha)
        elif self.alpha_activation == "sigmoid":
            alpha = self.sigmoid(alpha*self.sigmoid_tightness)
        if self.softplus_rd:
            rd = self.softplus(rd)

        out["alpha"] = alpha.reshape(original_shape)
        out["rd"] = rd.reshape(original_shape)
        out["xyz_encoded"] = xyz_encoded.reshape(original_shape)
        return out

    def get_params(self, lr):
        params = [
            {"params": self.encode_xyz.parameters(), "lr": lr},
            {"params": self.encode_angle.parameters(), "lr": lr},
            {"params": self.xyz_net.parameters(), "lr": lr},
            {"params": self.alpha_net.parameters(), "lr": lr},
            {"params": self.rd_net.parameters(), "lr": lr}
        ]
        return params