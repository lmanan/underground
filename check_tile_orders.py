import numpy as np
import zarr


# Setting directories
# root_dir = '/mnt/efs/shared_data/instance_no_gt/20230830_TIF_cellpose_test/'
# cellpose_pred_tiles_files = sorted(glob(os.path.join(root_dir, 'masks', '*.tif')))
# raw_tiles = sorted(glob(os.path.join(root_dir, 'raw_files', '*.tif')))


# image_test = tifffile.imread(raw_tiles[599])
# tile_shape = image_test.shape

tile_shape = (10, 20)

# getting the full shape
tiles_y = 26
tiles_x = 37
# tiles_x * tiles_y == len(raw_tiles) # True !

# computing big shape
big_shape = (tiles_x * tile_shape[0], tiles_y * tile_shape[1])
print(big_shape)  # 37440, 71040 pixel image; why we need .zarr


# define PATH
# path_to_stitch = '/mnt/efs/shared_data/instance_no_gt/big_zarr/stitched_image_plane_6.zarr'
path_to_stitch = './stitched.zarr'

# create new .zarr
zarrfile = zarr.open(path_to_stitch, mode='a')
zarr_image = zarrfile.require_dataset('image', shape=big_shape, chunks=tile_shape, dtype="uint16")


# In[38]:


# Defining sub-structures of the .zarr file

# First, need to arrange the tiles:

n_tiles = tiles_x * tiles_y
for idx in range(n_tiles):
    tile_index_y = idx // tiles_x
    tile_index_x = idx % tiles_x
    tile_index_y, tile_index_x = tile_index_x, tile_index_y

    # image = tifffile.imread(image)
    image = np.full(tile_shape, idx)

    # translate tile indices to actual coordinate
    index_y = tile_index_y * tile_shape[0]
    index_x = tile_index_x * tile_shape[1]
    # print(idx, ":", tile_index_y, tile_index_x, ":", index_y, index_x)
    # continue

    # time to write to .zarr
    zarr_image[index_y:(image.shape[0]+index_y), index_x:(image.shape[1]+index_x)] = image

# looking at it via napari
import napari
image = zarrfile['image']
napari.view_labels(image)
napari.run()
