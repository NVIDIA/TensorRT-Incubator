import tripy as tp
import torch
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TPBatchNorm(tp.Module):
    def __init__(self, num_features, eps=1e-5):
        super(TPBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Initialize learnable parameters (gamma and beta)
        self.gamma = tp.ones((num_features,), dtype=tp.float32)
        self.beta = tp.zeros((num_features,), dtype=tp.float32)

    def __call__(self, x):
        # Calculate mean and variance across the batch and spatial dimensions
        mean = tp.mean(x, dim=(0, 2, 3), keepdim=True)
        variance = tp.var(x, dim=(0, 2, 3), keepdim=True)

        # Normalize the input
        x_normalized = (x - mean) / tp.sqrt(variance + self.eps)

        # Apply the learned scaling (gamma) and shifting (beta)
        x_scaled = self.gamma * x_normalized + self.beta
        return x_scaled


class TPIdentity(tp.Module):
    def __init__(self):
        super(TPIdentity, self).__init__()

    def __call__(self, x):
        return x

class TPResNetConvLayer(tp.Module):
    def __init__(self, in_channels, out_channels, kernel_dims, stride=(1, 1), padding=((0, 0), (0, 0)), activation=True):
        super(TPResNetConvLayer, self).__init__()
        # All parameters should be passed as kernel_dims, stride, and padding in correct shape
        self.convolution = tp.Conv(
            in_channels, out_channels, kernel_dims=kernel_dims,
            stride=stride, padding=padding, bias=False
        )
        self.normalization = TPBatchNorm(out_channels)
        self.activation = tp.relu if activation else TPIdentity()

    def __call__(self, x):
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class TPResNetShortCut(tp.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(TPResNetShortCut, self).__init__()
        self.convolution = tp.Conv(
            in_channels, out_channels, kernel_dims=(1, 1),
            stride=stride, bias=False
        )
        self.normalization = TPBatchNorm(out_channels)

    def __call__(self, x):
        x = self.convolution(x)
        x = self.normalization(x)
        return x

class TPResNetBottleNeckLayer(tp.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels, stride):
        super(TPResNetBottleNeckLayer, self).__init__()
        self.shortcut = TPResNetShortCut(in_channels, out_channels, stride) if in_channels != out_channels or stride != (1, 1) else TPIdentity()

        self.conv1 = TPResNetConvLayer(in_channels, bottleneck_channels, kernel_dims=(1, 1), stride=(1, 1))
        self.conv2 = TPResNetConvLayer(bottleneck_channels, bottleneck_channels, kernel_dims=(3, 3), stride=stride, padding=((1, 1), (1, 1)))
        self.conv3 = TPResNetConvLayer(bottleneck_channels, out_channels, kernel_dims=(1, 1), stride=(1, 1), activation=False)
        self.activation = tp.relu

    def __call__(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        out = self.activation(out)
        return out

class TPResNetStage(tp.Module):
    def __init__(self, num_layers, in_channels, out_channels, bottleneck_channels, stride):
        super(TPResNetStage, self).__init__()
        self.layers = []
        for i in range(num_layers):
            layer = TPResNetBottleNeckLayer(
                in_channels if i == 0 else out_channels,
                out_channels, bottleneck_channels,
                stride if i == 0 else (1, 1)
            )
            setattr(self, f'layer_{i}', layer)
            self.layers.append(layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TPResNetEncoder(tp.Module):
    def __init__(self, layers_config):
        super(TPResNetEncoder, self).__init__()
        self.stages = []
        in_channels = 64
        for idx, (num_layers, out_channels, bottleneck_channels, stride) in enumerate(layers_config):
            stage = TPResNetStage(num_layers, in_channels, out_channels, bottleneck_channels, stride)
            setattr(self, f'stage_{idx}', stage)
            self.stages.append(stage)
            in_channels = out_channels

    def __call__(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

class TPMaxPool2d(tp.Module):
    def __init__(self, kernel_dims, stride=None, padding=((0, 0), (0, 0))):
        """
        Custom MaxPool2d class.

        :param kernel_dims: Size of the window to take the max over.
        :param stride: Stride of the window. If None, it will default to the kernel_dims.
        :param padding: Implicit padding added on both sides of the input. Should be a tuple of tuples.
        """
        super(TPMaxPool2d, self).__init__()
        self.kernel_dims = kernel_dims
        self.stride = stride if stride is not None else kernel_dims
        self.padding = padding

    def __call__(self, x):
        return tp.maxpool(x, kernel_dims=self.kernel_dims, stride=self.stride, padding=self.padding)

class TPResNetEmbeddings(tp.Module):
    def __init__(self):
        super(TPResNetEmbeddings, self).__init__()
        self.embedder = TPResNetConvLayer(3, 64, kernel_dims=(7, 7), stride=(2, 2), padding=((3, 3), (3, 3)))
        self.pooler = TPMaxPool2d(kernel_dims=(3, 3), stride=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x = self.embedder(x)
        x = self.pooler(x)
        return x

class TPAdaptiveAvgPool2d(tp.Module):
    def __init__(self, output_size):
        super(TPAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def __call__(self, x):
        N, C, H_in, W_in = x.shape
        H_out, W_out = self.output_size

        # Calculate stride and kernel size
        stride_h = H_in // H_out
        stride_w = W_in // W_out

        kernel_size_h = H_in - (H_out - 1) * stride_h
        kernel_size_w = W_in - (W_out - 1) * stride_w

        return tp.avgpool(x, kernel_dims=(int(kernel_size_h), int(kernel_size_w)), stride=(int(stride_h), int(stride_w)))

class TPResNetModel(tp.Module):
    def __init__(self):
        super(TPResNetModel, self).__init__()
        self.embedder = TPResNetEmbeddings()
        layers_config = [
            (3, 256, 64, (1, 1)),
            (4, 512, 128, (2, 2)),
            (6, 1024, 256, (2, 2)),
            (3, 2048, 512, (2, 2)),
        ]
        self.encoder = TPResNetEncoder(layers_config)
        self.pooler = TPAdaptiveAvgPool2d(output_size=(1, 1))

    def __call__(self, x):
        x = self.embedder(x)
        x = self.encoder(x)
        x = self.pooler(x)
        return x


logging.info("CUDA visible devices are: ")
for i in range(torch.cuda.device_count()):
    logging.info(f"\tDevice {i}: {torch.cuda.get_device_name(i)}")

x = tp.ones([1, 3, 224, 224], dtype=tp.float32)
model = TPResNetModel()

print(x)

logging.info("Model forward path running ")
start = time.perf_counter()

eager_output = model(x) 
logging.info(f"Forward path took {time.perf_counter() - start} seconds.")

logging.info("Model compilation running ")
input_shape = [1, 3, 224, 224]

compile_start_time = time.perf_counter()
model = tp.compile(model, args=[tp.InputInfo(input_shape, dtype=tp.float32)])
compile_end_time = time.perf_counter()
logging.info(f"Compilation took {compile_end_time - compile_start_time} seconds.")

logging.info("Compiled model forward path running ")
start = time.perf_counter()

eager_output = model(x) 
logging.info(f"Forward path for compiled model took {time.perf_counter() - start} seconds.")