import torch.nn.functional as F


def fusion_loss(rgb_seg_1,depth_add,label):

    _feature_map_B, _feature_map_C, feature_map_W, feature_map_H = rgb_seg_1.size()
    _label_B, label_W, label_H = label.size()
    if feature_map_W != label_W or feature_map_H != label_H:
        label = label.unsqueeze(1).float()
        label = F.interpolate(label,[feature_map_W,feature_map_H],mode="nearest")
        label = label.squeeze(1).long()
    loss_pr_seg = F.cross_entropy(rgb_seg_1, label)
    rgb_seg_1 = rgb_seg_1.detach()
    rgb_seg_1=rgb_seg_1.argmax(1)
    rgb_seg_1.eq_(label)
    rgb_seg_1=rgb_seg_1.clone().detach_().requires_grad_(False)
    add_map = (1-rgb_seg_1)*label
    add_map=add_map.clone().detach_().requires_grad_(False)

    loss_add = F.cross_entropy(depth_add,add_map)

    return loss_pr_seg,loss_add,add_map
