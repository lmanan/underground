import numpy as np
from cellpose import models
from glob import glob
import tifffile

filenames = sorted(glob('/mnt/efs/shared_data/instance_no_gt/Keyence_20230331_40x_DAPI_highres_zstack_acquisition_tile_scan/2DPinholeW10z22pitch1.3/XY01/IF_XY01_00599_Z*_CH4.tif'))
stack = []

for filename in filenames:
	im = tifffile.imread(filename)
	stack.append(im)

stack = np.asarray(stack)

gt_model = '/mnt/efs/shared_data/instance_no_gt/github/nuclei_20230830_HITL_3_final_model'

diameter = None
min_size = 0


model = models.Cellpose(gpu=True, pretrained_model=gt_model, net_avg=False)
masks, _, _, _ = model.eval(stack, diameter = nuclei_diam, min_size = min_size, stitch_threshold = 0.8)

tifffile.imwrite('IF_XY01_00599_masks.tif', masks)