from tripy.flat_ir.ops.base import BaseFlatIROp
from tripy.flat_ir.ops.add import AddOp
from tripy.flat_ir.ops.pow import PowOp
from tripy.flat_ir.ops.mul import MulOp
from tripy.flat_ir.ops.compare import CompareOp
from tripy.flat_ir.ops.constant import ConstantOp
from tripy.flat_ir.ops.copy import CopyOp
from tripy.flat_ir.ops.broadcast import BroadcastOp, DynamicBroadcastOp
from tripy.flat_ir.ops.iota import IotaOp
from tripy.flat_ir.ops.transpose import TransposeOp
from tripy.flat_ir.ops.select import SelectOp
from tripy.flat_ir.ops.slice import SliceOp
from tripy.flat_ir.ops.shape import ShapeOp
from tripy.flat_ir.ops.reshape import ReshapeOp
from tripy.flat_ir.ops.sub import SubtractOp
from tripy.flat_ir.ops.divide import DivideOp
from tripy.flat_ir.ops.reduce import ReduceOp, ArgMinMaxOp
from tripy.flat_ir.ops.exponential import ExpOp
from tripy.flat_ir.ops.tanh import TanhOp
from tripy.flat_ir.ops.gather import GatherOp
from tripy.flat_ir.ops.convert import ConvertOp
from tripy.flat_ir.ops.rsqrt import RsqrtOp
from tripy.flat_ir.ops.random_normal import RandomNormalOp
from tripy.flat_ir.ops.random_uniform import RandomUniformOp
from tripy.flat_ir.ops.maximum import MaxOp
from tripy.flat_ir.ops.minimum import MinOp
from tripy.flat_ir.ops.concatenate import ConcatenateOp
from tripy.flat_ir.ops.log import LogOp
