#----------------------------------------------------------------------------
# Created By  : Victoria BRAMI   
# Created Date: 2022/07/time ..etc
# version ='1.0'
# ---------------------------------------------------------------------------


from common.skeleton import Skeleton
from model.stmo_pretrain import Model_MAE
from common.opt_wholebody import opts
import torch
from torchsummary import summary
import numpy as np

# Define Normal Pose Skel
# Define Wholebody Pose Skel


def main():
    opt = opts().parse()
    opt.n_joints= 85
    opt.out_joints = 85
    opt.frames = 27
    opt.layers = 4
    model = Model_MAE(opt)
    print(model)
    
    
    print("\n /////// MAE Model ///////")
    input_data_1 = torch.rand((160, 2, 27, 85, 1))
    input_data_2 = torch.ones((27)).long()
    spatial_mask = np.zeros((27, 85), dtype=bool)
    import random
    for k in range(27):
        ran = random.sample(range(0, 84), opt.spatial_mask_num)
        spatial_mask[k, ran] = True
    spatial_mask = torch.from_numpy(spatial_mask)
    summary(model, input_data_1, input_data_2, spatial_mask)
    print("\n /////// Trans Model ///////")



if __name__ == '__main__':
    main()