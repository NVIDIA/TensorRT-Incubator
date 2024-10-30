import time
import tripy as tp
import torch

from sam2.modeling.backbones import image_encoder, hieradet
from sam2.modeling.position_encoding import PositionEmbeddingSine

from tripy.logging import logger

logger.verbosity = "mlir"


############## trunk -- Hiera #############
def test_trunk():
    trunk = hieradet.Hiera(
        embed_dim=144,
        num_heads=2,
        stages=[2, 6, 36, 4],
        global_att_blocks=[23, 33, 43],
        window_pos_embed_bkg_spatial_size=[7, 7],
        window_spec=[8, 4, 16, 8],
    )
    trunk.generate_static_pos_embed()
    # trunk_inp = tp.ones((1, 3, 1024, 1024))
    # trunk_out = trunk(trunk_inp)
    # print(trunk_out[1])

    print("Start compiling trunk...")
    start = time.time()
    compiled_tp_trunk = tp.compile(
        trunk,
        optimization_level=3,
        args=[
            tp.InputInfo((1, 3, 1024, 1024), dtype=tp.float32),
        ],
    )
    print(f"Compile trunk took {time.time() - start}s")


############## neck -- FpnNeck #############
def test_neck():
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )
    neck = image_encoder.FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=[1152, 576, 288, 144],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )

    # neck_inp = [
    #     tp.ones([1, 144, 256, 256]),
    #     tp.ones([1, 288, 128, 128]),
    #     tp.ones([1, 576, 64, 64]),
    #     tp.ones([1, 1152, 32, 32]),
    # ]
    # neck_out = neck(*neck_inp)
    # print(neck_out[3])

    print("Start compiling FpnNeck...")
    start = time.time()
    compiled_tp_neck = tp.compile(
        neck,
        optimization_level=3,
        args=[
            tp.InputInfo((1, 144, 256, 256), dtype=tp.float32),
            tp.InputInfo((1, 288, 128, 128), dtype=tp.float32),
            tp.InputInfo((1, 576, 64, 64), dtype=tp.float32),
            tp.InputInfo((1, 1152, 32, 32), dtype=tp.float32),
        ],
    )
    print(f"Compile image encoder took {time.time() - start}s")


############ image_encoder (trunk + neck) ##################
def test_image_encoder():
    trunk = hieradet.Hiera(
        embed_dim=144,
        num_heads=2,
        stages=[2, 6, 36, 4],
        global_att_blocks=[23, 33, 43],
        window_pos_embed_bkg_spatial_size=[7, 7],
        window_spec=[8, 4, 16, 8],
    )
    trunk.generate_static_pos_embed((256, 256))

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )
    neck = image_encoder.FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=[1152, 576, 288, 144],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )

    encoder = image_encoder.ImageEncoder(
        trunk=trunk,
        neck=neck,
        scalp=1,
    )

    # test eager mode
    # inp = tp.ones((1, 3, 1024, 1024))
    # out = encoder(inp)
    # print(out)

    # test compilation
    print("Start compiling image encoder...")
    start = time.time()
    compiled_tp_image_encoder = tp.compile(
        encoder.forward,
        args=[
            tp.InputInfo((1, 3, 1024, 1024), dtype=tp.float32),
        ],
    )
    print(f"Compile image encoder took {time.time() - start}s")


test_image_encoder()
