import nifty.tools as nt
import pandas as pd
import zarr

from skimage.measure import regionprops
from tqdm import trange


# TODO parallelize and merge this into mobie.utils
def compute_table_for_large_image(segmentation_path, segmentation_key, table_path):
    zarr_file = zarr.open(segmentation_path, "r")
    seg = zarr_file[segmentation_key]
    chunks = seg.chunks

    blocking = nt.tools([0, 0], seg.shape, chunks)
    tables = []

    for block_id in trange(blocking.numberOfBlocks):
        block = blocking.getBlock(block_id)
        offset = block.begin
        bb = tuple(slice(off, off + end) for off, end in zip(offset, block.end))

        tile_seg = seg[bb]
        props = regionprops(tile_seg)

        table_data = {
            "label_id": [prop.label for prop in props],
            "anchor_x": [prop.centroid[1] for prop in props],
            "anchor_y": [prop.centroid[0] for prop in props],
        }

        tables.append(table_data)

    table = pd.concat(tables)
    table.to_csv(table_path, index=False)


def main():
    seg_path = ""
    seg_key = ""
    table_path = ""
    compute_table_for_large_image(seg_path, seg_key, table_path)


if __name__ == "__main__":
    main()
