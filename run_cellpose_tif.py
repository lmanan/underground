import numpy as np
from cellpose import models
from glob import glob
import tifffile
from tqdm import tqdm

filenames = sorted(glob('/mnt/efs/shared_data/instance_no_gt/Keyence_20230331_40x_DAPI_highres_zstack_acquisition_tile_scan/2DPinholeW10z22pitch1.3/XY01/*.tif'))
gt_model = '/mnt/efs/shared_data/instance_no_gt/github/nuclei_20230830_HITL_3_final_model'
model = models.CellposeModel(gpu=True, pretrained_model=gt_model, net_avg=False)

diameter = None
min_size = 0
stitch_threshold = 0.4

for tile in tqdm(962):
	stack = []
	for z_slice in range(18):
		im = tifffile.imread(filenames[(z_slice)+18*(tile)])
		stack.append(im)
	stack = np.asarray(stack)
	masks, _, _ = model.eval(stack, diameter = diameter, min_size = min_size, stitch_threshold = stitch_threshold)
	tifffile.imwrite(f'/mnt/efs/shared_data/instance_no_gt/20230830_TIF_cellpose_test/raw_files/IF_XY01_{str(tile+1)}_raw.tif', stack)
	tifffile.imwrite(f'/mnt/efs/shared_data/instance_no_gt/20230830_TIF_cellpose_test/masks/IF_XY01_{str(tile+1)}_masks.tif', masks)






