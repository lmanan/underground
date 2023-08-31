import numpy as np
from cellpose import models
from glob import glob
import tifffile
from tqdm import tqdm
import os
import zarr

filenames = sorted(glob('/mnt/efs/shared_data/instance_no_gt/Keyence_20230331_40x_DAPI_highres_zstack_acquisition_tile_scan/2DPinholeW10z22pitch1.3/XY01/IF_XY01_*_Z006_CH4.tif'))
gt_model = '/mnt/efs/shared_data/instance_no_gt/github/nuclei_20230830_HITL_3_final_model'
model = models.CellposeModel(gpu=True, pretrained_model=gt_model, net_avg=False)

diameter = None
min_size = 200
stitch_threshold = 0.4
for filename in tqdm(filenames):
    file_data = str.split(os.path.basename(filename), '_')
    image = tifffile.imread(filename)
    image = np.asarray(image)
    masks,_,_=model.eval(image, diameter=diameter, min_size=min_size)
    tifffile.imwrite('/mnt/efs/shared_data/instance_no_gt/20230830_TIF_cellpose_test/raw_files/raw_'+os.path.basename(filename), image)
    tifffile.imwrite('/mnt/efs/shared_data/instance_no_gt/20230830_TIF_cellpose_test/masks/masks_'+os.path.basename(filename), masks)
