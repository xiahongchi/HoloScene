import torch
from torch import nn
import utils.general as utils
import math
import torch.nn.functional as F

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def dilate_mask(mask, k=1):
    """
    Dilate a binary mask by k pixels

    Args:
        mask (torch.Tensor): Binary mask of shape (1, H, W) or (B, 1, H, W)
        k (int): Number of pixels to dilate by

    Returns:
        torch.Tensor: Dilated mask with same shape as input
    """
    # Ensure mask has 4 dimensions (B, C, H, W)
    mask_shape = mask.shape
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # Add batch dimension if missing

    # Create a (2k+1) x (2k+1) kernel filled with ones
    kernel_size = 2 * k + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    # Perform the dilation using convolution
    # The padding ensures the output has the same size as the input
    dilated_mask = F.conv2d(mask.float(), kernel, padding=k) > 0

    # Return tensor with the same shape as the input
    return dilated_mask.to(mask.dtype).reshape(*mask_shape)


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


# def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

#     M = torch.sum(mask, (1, 2))
#     res = prediction - target
#     image_loss = torch.sum(mask * res * res, (1, 2))

#     return reduction(image_loss, 2 * M)

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    res = prediction - target
    image_loss = mask * res * res

    return image_loss


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        # self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        # self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        # if self.__alpha > 0:
        #     total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy

def compute_scale_and_shift_batch(prediction, target):
    # prediction: (B, N), N = 32
    # target: (B, N)
    B, N = prediction.shape
    dr = prediction.unsqueeze(-1) # (B, N, 1)
    dr = torch.cat((dr, torch.ones_like(dr).to(dr.device)), dim=-1).reshape(-1, 2, 1)  # (BxN, 2, 1)
    dr_sq = torch.sum((dr @ dr.transpose(1, 2)).reshape(B, N, 2, 2), dim=1) # (B, 2, 2)
    # left_part = torch.inverse(dr_sq+torch.eye(2).reshape(1, 2, 2).cuda()*1e-6).reshape(B, 2, 2) # (B, 2, 2)
    left_part = torch.inverse(dr_sq).reshape(B, 2, 2) # (B, 2, 2)
    right_part = torch.sum((dr.reshape(B, N, 2, 1))*(target.reshape(B, N, 1, 1)), dim=1).reshape(B, 2, 1)
    rs = left_part @ right_part # (B, 2, 1)
    rs = rs.reshape(B, 2)
    return rs[:, 0], rs[:, 1]


class MonoSDFLoss(nn.Module):
    def __init__(self, rgb_loss, 
                 eikonal_weight, 
                 smooth_weight = 0.005,
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 uncertainty_begin_iter = 20000000,           # set a large number to avoid using uncertainty
                 depth_type = 'marigold',
                 phy_un_weight = 50,
                 end_step = -1):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.smooth_weight = smooth_weight
        self.depth_weight = depth_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.uncertainty_begin_iter = uncertainty_begin_iter
        self.depth_type = depth_type
        self.phy_un_weight = phy_un_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')

        self.use_uncertainty = True
        
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        
        # print(f"using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = end_step

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self,model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']
        
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss =  torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss
    
    def get_depth_loss(self, depth_pred, depth_gt):
        # TODO remove hard-coded scaling for depth

        # if self.depth_type == 'marigold':
        #     depth_loss = self.depth_loss(depth_pred.reshape(1, -1, 32), (depth_gt * 4.0 + 0.5).reshape(1, -1, 32), mask.reshape(1, -1, 32))
        # elif self.depth_type == 'omnidata':
        #     depth_loss = self.depth_loss(depth_pred.reshape(1, -1, 32), (depth_gt * 50.0 + 0.5).reshape(1, -1, 32), mask.reshape(1, -1, 32))
        # else:
        #     raise ValueError(f'{self.depth_type} not implement')
        # print("mask: ", mask.shape, torch.count_nonzero(mask))
        # print("depth_pred: ", depth_pred.shape)
        # print("depth_gt: ", depth_gt.shape)
        #
        # assert False

        # mask = mask.reshape(-1)

        depth_pred = depth_pred.reshape(-1).reshape(1, -1)
        depth_gt = depth_gt.reshape(-1).reshape(1, -1)
        w, q = compute_scale_and_shift_batch(depth_pred, depth_gt)

        w = w.reshape(-1, 1)
        q = q.reshape(-1, 1)

        diff = ((w * depth_pred + q) - depth_gt) ** 2
        depth_loss = torch.clip(diff, max=1)

        depth_loss = depth_loss.reshape(-1)

        depth_loss = depth_loss.mean()

        return depth_loss
        
    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)
        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1))

        l1 = torch.mean(l1)
        cos = torch.mean(cos)

        return l1, cos
        
    def forward(self, model_outputs, ground_truth):
        
        # import pdb; pdb.set_trace()
        rgb_gt = ground_truth['rgb'].cuda()
        # monocular depth and normal
        depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()
        
        depth_pred = model_outputs['depth_values']
        normal_pred = model_outputs['normal_map'][None]

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        # only supervised the foreground normal
        mask = ((model_outputs['sdf'] > 0.).any(dim=-1) & (model_outputs['sdf'] < 0.).any(dim=-1))[None, :, None]
        # combine with GT
        mask = (ground_truth['mask'] > 0.5).cuda() & mask

        depth_loss = self.get_depth_loss(depth_pred, depth_gt) if self.depth_weight > 0 else torch.tensor(0.0).cuda().float()
        if isinstance(depth_loss, float):
            depth_loss = torch.tensor(0.0).cuda().float()    
        
        normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt)
        
        smooth_loss = self.get_smooth_loss(model_outputs)
        
        # compute decay weights 
        if self.end_step > 0:
            decay = math.exp(-self.step / self.end_step * 10.)
        else:
            decay = 1.0
            
        self.step += 1

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss +\
               self.smooth_weight * smooth_loss +\
               decay * self.depth_weight * depth_loss +\
               decay * self.normal_l1_weight * normal_l1 +\
               decay * self.normal_cos_weight * normal_cos               
        
        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'smooth_loss': smooth_loss,
            'depth_loss': depth_loss,
            'normal_l1': normal_l1,
            'normal_cos': normal_cos
        }

        return output


class HoloSceneLoss(MonoSDFLoss):
    def __init__(self, rgb_loss, 
                 eikonal_weight,
                 semantic_weight = 0.04,
                 smooth_weight = 0.005,
                 semantic_loss = torch.nn.CrossEntropyLoss(ignore_index = -1),
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 reg_vio_weight = 0.1,
                 use_obj_opacity = True,
                 bg_reg_weight = 0.1,
                 depth_type = 'marigold',
                 end_step = -1):
        super().__init__(
                 rgb_loss = rgb_loss, 
                 eikonal_weight = eikonal_weight, 
                 smooth_weight = smooth_weight,
                 depth_weight = depth_weight,
                 normal_l1_weight = normal_l1_weight,
                 normal_cos_weight = normal_cos_weight,
                 depth_type = depth_type,
                 end_step = end_step)
        self.semantic_weight = semantic_weight
        self.bg_reg_weight = bg_reg_weight

        self.semantic_loss = utils.get_class(semantic_loss)(reduction='none') if semantic_loss is not torch.nn.CrossEntropyLoss else torch.nn.CrossEntropyLoss(ignore_index = -1, reduction='none')
        self.reg_vio_weight = reg_vio_weight
        self.use_obj_opacity = use_obj_opacity

        print(f"[INFO]: using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}\
            Semantic_{self.semantic_weight}, semantic_loss_type_{self.semantic_loss} Use_object_opacity_{self.use_obj_opacity} Reg_vio_{self.reg_vio_weight} BG_reg_{self.bg_reg_weight} Depth_type_{depth_type}")

    def get_semantic_loss(self, semantic_value, semantic_gt):
        semantic_gt = semantic_gt.squeeze()
        semantic_loss = self.semantic_loss(semantic_value, semantic_gt)

        semantic_loss = torch.mean(semantic_loss)
        return semantic_loss
    
    def object_distinct_loss(self, sdf_value, min_sdf):
        _, min_indice = torch.min(sdf_value, dim=1, keepdims=True)
        input = -sdf_value - min_sdf.detach()
        input = torch.relu(input)
        N, K = input.shape
        mask = torch.ones((N, K), dtype=torch.bool, device=input.device)
        min_indice = min_indice.reshape(-1)
        mask[torch.arange(N).long().cuda(), min_indice] = False
        input = input[mask]
        input = input.reshape(-1)
        collision_cnt = torch.count_nonzero(input > 0)
        if collision_cnt > 0:
            loss = input.sum() / collision_cnt
        else:
            loss = torch.tensor(0.0).cuda().float()
        return loss

    def object_distinct_graph_loss(self, sdf_value, collision_relations):

        N, K = sdf_value.shape

        self_id = collision_relations["obj_i"]
        parent_id = collision_relations["parent"]
        desc_ids = collision_relations["desc"]
        bothers = collision_relations["bother"]
        scale = collision_relations["scale"]
        offset = 0. * scale

        has_parent = parent_id >= 0
        has_desc = len(desc_ids) > 0
        has_bother = len(bothers) > 0

        parent_loss = torch.tensor(0.0).cuda().float()
        if has_parent:
            selected_indices = [parent_id, self_id, *desc_ids]
            sdf_value_selected = sdf_value[:, selected_indices]
            sdf_value_selected = sdf_value_selected.reshape(N, -1)
            inside_mask = sdf_value_selected[:, 0] < 0
            parent_intersect = -sdf_value_selected[inside_mask, 1:] - sdf_value_selected[inside_mask, 0:1].detach() + offset
            parent_intersect_mask = parent_intersect > 0
            parent_collision_cnt = torch.count_nonzero(parent_intersect_mask)
            if parent_collision_cnt > 0:
                parent_loss = parent_intersect[parent_intersect_mask].sum() / parent_collision_cnt
            else:
                parent_loss = torch.tensor(0.0).cuda().float()

        desc_loss = torch.tensor(0.0).cuda().float()
        if has_desc:
            selected_indices = [self_id, *desc_ids]
            sdf_value_selected = sdf_value[:, selected_indices]
            sdf_value_selected = sdf_value_selected.reshape(N, -1)
            inside_mask = sdf_value_selected[:, 0] < 0
            desc_intersect = -sdf_value_selected[inside_mask, 1:] - sdf_value_selected[inside_mask, 0:1].detach() + offset
            desc_intersect_mask = desc_intersect > 0
            desc_collision_cnt = torch.count_nonzero(desc_intersect_mask)
            if desc_collision_cnt > 0:
                desc_loss = desc_intersect[desc_intersect_mask].sum() / desc_collision_cnt
            else:
                desc_loss = torch.tensor(0.0).cuda().float()

        bother_loss = torch.tensor(0.0).cuda().float()
        if has_bother:
            self_and_desc_ids = [self_id, *desc_ids]
            self_sdf_value = sdf_value[:, self_and_desc_ids]
            self_sdf_value = self_sdf_value.reshape(N, -1)
            min_self_sdf_value = torch.min(self_sdf_value, dim=1)[0].reshape(N, 1)
            sdf_compare = min_self_sdf_value

            for bother in bothers:
                sdf_value_bother = sdf_value[:, bother]
                sdf_value_bother = sdf_value_bother.reshape(N, -1)
                min_sdf_value_bother = torch.min(sdf_value_bother, dim=1)[0]
                sdf_compare = torch.cat((sdf_compare, min_sdf_value_bother.reshape(N, 1)), dim=1)

            min_sdf_value, min_indice = torch.min(sdf_compare, dim=1, keepdims=True)
            min_sdf_value = min_sdf_value.reshape(N, 1)
            inside_mask = min_sdf_value[:, 0] < 0
            input = -sdf_compare[inside_mask] - min_sdf_value[inside_mask].detach() + offset

            N, K_bothers = input.shape
            mask = torch.ones((N, K_bothers), dtype=torch.bool, device=input.device)
            min_indice = min_indice.reshape(-1)[inside_mask]
            mask[torch.arange(N).long().cuda(), min_indice] = False

            input = input[mask]

            input = input.reshape(-1)

            collision_cnt = torch.count_nonzero(input > 0)
            input = torch.relu(input)
            if collision_cnt > 0:
                bother_loss = input.sum() / collision_cnt
            else:
                bother_loss = torch.tensor(0.0).cuda().float()

        return parent_loss, desc_loss, bother_loss


    def object_opacity_loss(self, predict_opacity, gt_opacity, weight=None):
        target = torch.nn.functional.one_hot(gt_opacity.squeeze(), num_classes=predict_opacity.shape[1]).float()        # [ray_num, obj_num]
        predict_opacity = torch.clip(predict_opacity, 1e-4, 1-(1e-4))
        loss = F.binary_cross_entropy(predict_opacity, target, reduction='none').mean(dim=-1)    # [ray_num]
        loss = torch.mean(loss)
        return loss


    def get_bg_render_loss(self, bg_depth, bg_normal, mask):

        bg_depth = bg_depth.reshape(1, 32, 32)
        bg_normal = bg_normal.reshape(32, 32, 3).permute(2, 0, 1)

        mask = mask.reshape(1, 32, 32)

        depth_grad = self.compute_grad_error(bg_depth, mask)
        normal_grad = self.compute_grad_error(bg_normal, mask.repeat(3, 1, 1))

        bg_render_loss = depth_grad + normal_grad
        return bg_render_loss

    def get_bg_surface_obj_reg(self, obj_sdfs):
        margin_target = torch.ones(obj_sdfs.shape).cuda()
        threshold = 0.05 * torch.ones(obj_sdfs.shape).cuda()
        loss = torch.nn.functional.margin_ranking_loss(obj_sdfs, threshold, margin_target)

        return loss
    
    def compute_grad_error(self, x, mask):
        scales = 4
        grad_loss = torch.tensor(0.0).cuda().float()
        for i in range(scales):
            step = pow(2, i)

            mask_step = mask[:, ::step, ::step]
            x_step = x[:, ::step, ::step]

            M = torch.sum(mask_step[:1], (1, 2))

            diff = torch.mul(mask_step, x_step)

            grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
            mask_x = torch.mul(mask_step[:, :, 1:], mask_step[:, :, :-1])
            grad_x = torch.mul(mask_x, grad_x)

            grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
            mask_y = torch.mul(mask_step[:, 1:, :], mask_step[:, :-1, :])
            grad_y = torch.mul(mask_y, grad_y)

            image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

            divisor = torch.sum(M)

            if divisor == 0:
                scale_loss = torch.tensor(0.0).cuda().float()
            else:
                scale_loss = torch.sum(image_loss) / divisor

            grad_loss += scale_loss

        return grad_loss
    
    def compute_grad_error_triplet(self, x, mask_p, mask_d):

        scales = 4
        grad_loss = torch.tensor([0.0]).cuda().float()
        for i in range(scales):
            step = pow(2, i)

            mask_p_step = mask_p[:, ::step, ::step]
            mask_d_step = mask_d[:, ::step, ::step]
            x_step = x[:, ::step, ::step]

            diff_p = torch.mul(mask_p_step, x_step).detach()
            diff_d = torch.mul(mask_d_step, x_step)

            grad_x = torch.abs(diff_p[:, :, 1:] - diff_d[:, :, :-1])
            mask_x = torch.mul(mask_p_step[:, :, 1:], mask_d_step[:, :, :-1])
            cnt_x = int(torch.sum(mask_x[:1]))
            grad_x = torch.mul(mask_x, grad_x)

            grad_y = torch.abs(diff_p[:, 1:, :] - diff_d[:, :-1, :])
            mask_y = torch.mul(mask_p_step[:, 1:, :], mask_d_step[:, :-1, :])
            cnt_y = int(torch.sum(mask_y[:1]))
            grad_y = torch.mul(mask_y, grad_y)
            if cnt_x == 0:
                scale_loss_x = torch.tensor(0.0).cuda().float()
            else:
                scale_loss_x = torch.sum(grad_x) / cnt_x

            if cnt_y == 0:
                scale_loss_y = torch.tensor(0.0).cuda().float()
            else:
                scale_loss_y = torch.sum(grad_y) / cnt_y

            grad_loss += (scale_loss_x.reshape(1) + scale_loss_y.reshape(1))


            # flip

            grad_x_f = torch.abs(diff_p[:, :, :-1] - diff_d[:, :, 1:])
            mask_x_f = torch.mul(mask_p_step[:, :, :-1], mask_d_step[:, :, 1:])
            cnt_x_f = int(torch.sum(mask_x_f[:1]))
            grad_x_f = torch.mul(mask_x_f, grad_x_f)

            grad_y_f = torch.abs(diff_p[:, :-1, :] - diff_d[:, 1:, :])
            mask_y_f = torch.mul(mask_p_step[:, :-1, :], mask_d_step[:, 1:, :])
            cnt_y_f = int(torch.sum(mask_y_f[:1]))
            grad_y_f = torch.mul(mask_y_f, grad_y_f)
            if cnt_x_f == 0:
                scale_loss_x_f = torch.tensor(0.0).cuda().float()
            else:
                scale_loss_x_f = torch.sum(grad_x_f) / cnt_x_f

            if cnt_y_f == 0:
                scale_loss_y_f = torch.tensor(0.0).cuda().float()
            else:
                scale_loss_y_f = torch.sum(grad_y_f) / cnt_y_f

            grad_loss += (scale_loss_x_f.reshape(1) + scale_loss_y_f.reshape(1))

        return grad_loss


    def forward(self, model_outputs, ground_truth, call_reg=False, call_bg_reg=False):
        output = super().forward(model_outputs, ground_truth)

        iter_step = model_outputs['iter_step']
        if 'semantic_values' in model_outputs and not self.use_obj_opacity: # ObjectSDF loss: semantic field + cross entropy
            semantic_gt = ground_truth['segs'].cuda().long()
            semantic_loss = self.get_semantic_loss(model_outputs['semantic_values'], semantic_gt)
        elif "object_opacity" in model_outputs and self.use_obj_opacity: # ObjectSDF++ loss: occlusion-awared object opacity + MSE
            semantic_gt = ground_truth['segs'].cuda().long()
            semantic_loss = self.object_opacity_loss(model_outputs['object_opacity'], semantic_gt)
        else:
            semantic_loss = torch.tensor(0.0).cuda().float()
        
        if "sample_sdf" in model_outputs and call_reg:
            # print("model_outputs: ", model_outputs.keys())
            if 'collision_relations' in model_outputs:
                # print("use object_distinct_graph_loss")
                parent_loss, desc_loss, bother_loss = self.object_distinct_graph_loss(
                    model_outputs["sample_sdf"],
                    model_outputs["collision_relations"]
                )
                sample_sdf_loss = parent_loss + desc_loss + bother_loss
                output['collision_reg_parent_loss'] = parent_loss
                output['collision_reg_desc_loss'] = desc_loss
                output['collision_reg_bother_loss'] = bother_loss

            else:
                # print("use object_distinct_loss")
                sample_sdf_loss = self.object_distinct_loss(model_outputs["sample_sdf"], model_outputs["sample_minsdf"])

        else:
            sample_sdf_loss = torch.tensor(0.0).cuda().float()

        # background_reg_loss = torch.tensor(0.0).cuda().float()        
        if 'bg_depth_values' in model_outputs:
            if 'bg_mask' in model_outputs:
                bg_mask = (model_outputs['bg_mask'] !=0).int()         # only smooth occluded background, i.e. semantic value is not 0
            else:
                bg_mask = (ground_truth['segs'] != 0).cuda()            # use gt mask directly
            background_reg_loss = self.get_bg_render_loss(model_outputs['bg_depth_values'], model_outputs['bg_normal_map'], bg_mask)
        else:
            background_reg_loss = torch.tensor(0.0).cuda().float()

        if 'rgb_offset' in model_outputs:
            rgb_offset_loss = torch.mean(model_outputs['rgb_offset'] ** 2)
            output['loss'] = output['loss'] + rgb_offset_loss
            output['rgb_offset_loss'] = rgb_offset_loss

        output['semantic_loss'] = semantic_loss
        output['collision_reg_loss'] = sample_sdf_loss
        output['background_reg_loss'] = background_reg_loss
        output['loss'] = output['loss'] + \
                         self.semantic_weight * semantic_loss + \
                         self.reg_vio_weight* sample_sdf_loss + \
                         self.bg_reg_weight * background_reg_loss
        return output