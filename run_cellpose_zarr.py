from cellpose import models



z_gt = zarr.open(
    ,
    mode='w',
    shape=(962,18,1440,1920),
    dtype=np.int32
) # Make new zarr

z_raw = zarr.open() # Load zarr file with raw data

our_model = # Write path to trained model
nuclei_diam = # Write expected nucleus diameter

for tile in range(z_raw.shape[0]):
    z[i] = np.load(fn)
    model = models.Cellpose(gpu=True, pretrained_model=our_model, net_avg=False, stitch_threshold = 0.8)
    masks, flows, styles, diams = model.eval(z_raw[tile], batch_size = 8, diameter = nuclei_diam, min_size=min_size, stitch_threshold = 0.8)