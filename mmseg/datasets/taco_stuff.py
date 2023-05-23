# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class TACOStuffDataset(BaseSegDataset):
    """TACO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=(['Aluminium foil', 'Battery', 'Aluminium blister pack', 'Carded blister pack',
		 'Other plastic bottle', 'Clear plastic bottle', 'Glass bottle', 'Plastic bottle cap',
		 'Metal bottle cap', 'Broken glass', 'Food Can', 'Aerosol',
		 'Drink can', 'Toilet tube', 'Other carton', 'Egg carton',
		 'Drink carton', 'Corrugated carton', 'Meal carton', 'Pizza box',
		 'Paper cup', 'Disposable plastic cup', 'Foam cup', 'Glass cup',
		 'Other plastic cup', 'Food waste', 'Glass jar', 'Plastic lid',
		 'Metal lid', 'Other plastic', 'Magazine paper', 'Tissues',
		 'Wrapping paper', 'Normal paper', 'Paper bag', 'Plastified paper bag',
		 'Plastic film', 'Six pack rings', 'Garbage bag', 'Other plastic wrapper',
		 'Single-use carrier bag', 'Polypropylene bag', 'Crisp packet', 'Spread tub',
		 'Tupperware', 'Disposable food container', 'Foam food container', 'Other plastic container',
		 'Plastic glooves', 'Plastic utensils', 'Pop tab', 'Rope & strings',
 		 'Scrap metal', 'Shoe', 'Squeezable tube', 'Plastic straw',
		 'Paper straw', 'Styrofoam piece', 'Unlabeled litter', 'Cigarette'], 
        palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                 [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                 [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                 [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                 [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                 [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                 [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                 [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                 [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                 [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                 [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                 [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                 [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                 [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
