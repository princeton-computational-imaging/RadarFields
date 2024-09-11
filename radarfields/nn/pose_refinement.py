# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import torch
from torch import nn
import numpy as np
from scipy.interpolate import interp1d

def exp_map_SO3xR3(tangent_vector):
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    Float[Tensor, "b 6"] -> Float[Tensor, "b 3 4"]

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret

def exp_map_SE3(tangent_vector):
    """Compute the exponential map `se(3) -> SE(3)`.

    Float[Tensor, "b 6"] -> Float[Tensor, "b 3 4"]

    This can be used for learning pose deltas on `SE(3)`.

    Args:
        tangent_vector: A tangent vector from `se(3)`. 

    Returns:
        [R|t] transformation matrices.
    """

    tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
    tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

    theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
    theta2 = theta**2
    theta3 = theta**3

    near_zero = theta < 1e-2
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    theta3_nz = torch.where(near_zero, non_zero, theta3)

    # Compute the rotation
    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz)
    ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)

    ret[:, 0, 0] += cosine.view(-1)
    ret[:, 1, 1] += cosine.view(-1)
    ret[:, 2, 2] += cosine.view(-1)
    temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
    ret[:, 0, 1] -= temp[:, 2]
    ret[:, 1, 0] += temp[:, 2]
    ret[:, 0, 2] += temp[:, 1]
    ret[:, 2, 0] -= temp[:, 1]
    ret[:, 1, 2] -= temp[:, 0]
    ret[:, 2, 1] += temp[:, 0]

    # Compute the translation
    sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2)
    theta_minus_sine_by_theta3_t = torch.where(near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz)

    ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
    ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(tangent_vector_ang, tangent_vector_lin, dim=1)
    ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
    )
    return ret

class PoseOptimizer(nn.Module):
    """## Layer that modifies camera poses to be optimized as well as the field during training."""
    def __init__(self, poses, mode, colinear, device, non_trainable_camera_indices=None):
        super().__init__()
        self.poses = poses
        self.num_cameras = poses.size(0)
        self.mode = mode
        self.colinear = colinear
        self.device = device
        if non_trainable_camera_indices is not None:
            self.non_trainable_camera_indices = torch.tensor(non_trainable_camera_indices, device=device)
        else: self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.mode == "off":
            pass
        elif self.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((self.num_cameras, 6), device=device)) # [N, 6]

    def forward(
        self,
        indices,
    ):
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = None

        if self.mode == "off":
            pass

        pose_adjustment = self.pose_adjustment.clone()

        # Detach non-trainable indices by zero-ing out adjustments
        if self.non_trainable_camera_indices is not None:
            pose_adjustment[self.non_trainable_camera_indices] = torch.zeros((1,6), device=self.pose_adjustment.device)

        # Apply learned transformation delta.
        if self.mode == "SO3xR3":
            outputs = exp_map_SO3xR3(pose_adjustment[indices, :])
        elif self.mode == "SE3":
            outputs = exp_map_SE3(pose_adjustment[indices, :])

        # Return identity if no transforms are needed
        if outputs is None:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return outputs

    def apply_to_poses(self, indices):
        """Apply the pose corrections to sensor poses"""
        if self.mode != "off":
            adj = self(indices)
            adj = torch.cat([adj, torch.Tensor([0, 0, 0, 1])[None, None].expand((adj.shape[0], 1, 4)).to(adj)], dim=1)
            return torch.bmm(self.poses[indices,...], adj)
        
    def get_regularization_penalty(self):
        """Pose gradient regularization penalty"""
        poses = self.get_corrected_poses()
        return torch.sum(torch.abs(torch.gradient(poses, dim=0)[0]))
        
    def get_coplanarity_penalty(self):
        """
        Compute a penalty that encourages the poses to be coplanar.
        
        Args:
            poses: A tensor of shape (N, 3) representing the 3D coordinates of the poses.

        Returns:
            penalty: A scalar tensor representing the coplanarity penalty.
        """

        poses = self.get_corrected_poses()
        rotations = poses[:, :3, :3]  # (N, 3, 3)
        translations = poses[:, :3, 3]  # (N, 3)
        N = poses.size(0)
        
        # Calculate the normal vectors of the planes defined by each pose matrix
        normal_vectors = torch.cross(rotations[:, :, 0], rotations[:, :, 1], dim=1)  # (N, 3)
        normal_vectors = normal_vectors / normal_vectors.norm(dim=1, keepdim=True)  # normalize the normals

        # Verify that all normal vectors are parallel by penalizing their pairwise dot product differences
        dot_products = torch.matmul(normal_vectors, normal_vectors.transpose(0, 1))
        normal_parallelism_penalty = ((1 - torch.abs(dot_products)).sum()) / 2

        # Ensure all translations are coplanar with respect to the plane defined by the first normal vector
        translation_plane_penalty = torch.abs((translations - translations.mean(dim=0)) @ normal_vectors[0]).sum()

        # Ensure all x-axis direction vectors are collinear
        collinearity_penalty = 0.0
        
        if self.colinear:
            x_axes = rotations[:, :, 0] # (N, 3)

            # Choose the first x-axis vector as the reference direction
            reference_x_axis = x_axes[0]
            reference_x_axis = reference_x_axis / reference_x_axis.norm()
            
            # Compute the dot products and penalize deviations from 1
            dot_products_x = torch.matmul(x_axes, reference_x_axis.unsqueeze(1)).squeeze()
            collinearity_penalty = (1 - torch.abs(dot_products_x)).sum()

        return normal_parallelism_penalty + translation_plane_penalty + collinearity_penalty

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_cameras).long())
    
    def get_corrected_poses(self):
        """Get all optimized pose correction matrices and apply them to all poses"""
        return self.apply_to_poses(torch.arange(1, self.num_cameras).long())
    
    def interp_test_poses(self, test_indices):
        """Interpolate between nearest optimized train poses to refine test frame poses"""

        with torch.no_grad():

            train_indices = np.array([i for i in range(self.num_cameras) if i not in test_indices])
            train_adjustment = self.pose_adjustment.clone().detach()[torch.tensor(train_indices), :]
            if train_adjustment.device == "cpu":
                train_adjustment = train_adjustment.numpy()
            else:
                train_adjustment = train_adjustment.cpu().numpy()

            interp_func = interp1d(train_indices, train_adjustment, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
            interpolated_test_adjustment = torch.tensor(interp_func(test_indices), device=self.device, dtype=torch.float32)

            new_adjustment = self.pose_adjustment.clone().detach().to(self.device)# [N, 6]
            new_adjustment[test_indices,:] = interpolated_test_adjustment
            self.pose_adjustment = torch.nn.Parameter(new_adjustment)
    
    def load_params(self, params):
        """Load in params from checkpoint"""
        self.pose_adjustment = torch.nn.Parameter(params, device=self.device)

    def get_metrics(self):
        """Get camera optimizer metrics"""
        if self.mode != "off":
            return self.pose_adjustment[:, :3].norm(), self.pose_adjustment[:, 3:].norm()

    def get_params(self, lr):
        """Get camera optimizer parameters"""
        assert(len(list(self.parameters())) > 0)
        params = [
            {"params": self.parameters(), "lr": lr},
        ]
        return params