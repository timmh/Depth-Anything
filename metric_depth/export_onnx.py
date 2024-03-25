# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat
# Modified by: Timm Haucke

import argparse
from pprint import pprint
import json

import torch
import onnx
import onnxruntime
import cv2
import numpy as np

from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import get_config


@torch.no_grad()
def infer(model, images, disable_flip=False, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)
    if disable_flip:
        return pred1
    else:
        pred2 = model(torch.flip(images, [3]), **kwargs)
        pred2 = get_depth_from_prediction(pred2)
        pred2 = torch.flip(pred2, [3])

        mean_pred = 0.5 * (pred1 + pred2)

        return mean_pred


@torch.no_grad()
def export(model, output_path, disable_flip, test_image_path):
    model.eval()

    class ModelWithFlipAug(torch.nn.Module):
        def __init__(self, model, disable_flip):
            super().__init__()
            self.model = model
            self.disable_flip = disable_flip

        def forward(self, *args, **kwargs):
            return infer(self.model, *args, **kwargs, disable_flip=disable_flip)

    
    net_w, net_h = 640, 480
    normalization = dict(mean=[0, 0, 0], std=[1, 1, 1])

    image = torch.zeros((1, 3, net_h, net_w)).to(model.device)
    model_with_flip_aug = ModelWithFlipAug(model, disable_flip)

    torch.onnx.export(
        model_with_flip_aug,
        image,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
    )

    model_onnx = onnx.load(output_path)
    meta_imagesize = model_onnx.metadata_props.add()
    meta_imagesize.key = "ImageSize"
    meta_imagesize.value = json.dumps([net_w, net_h])
    meta_normalization = model_onnx.metadata_props.add()
    meta_normalization.key = "Normalization"
    meta_normalization.value = json.dumps(normalization)
    meta_prediction_factor = model_onnx.metadata_props.add()
    meta_prediction_factor.key = "PredictionFactor"
    meta_prediction_factor.value = str(1)
    onnx.save(model_onnx, output_path)
    print(f"ONNX model saved as '{output_path}'")

    if test_image_path is None:
        print("Specify test image path to test for successful export")
    else:
        img = cv2.imread(test_image_path)

        # resize
        img_input = cv2.resize(img, (net_h, net_w), cv2.INTER_AREA)

        # normalize
        img_input = (img_input - np.array(normalization["mean"])) / np.array(normalization["std"])

        # transpose from HWC to CHW
        img_input = img_input.transpose(2, 0, 1)

        # add batch dimension
        img_input = np.stack([img_input])

        # validate accuracy of exported model
        torch_out = model_with_flip_aug(torch.from_numpy(img_input.astype(np.float32))).detach().cpu().numpy()
        session = onnxruntime.InferenceSession(
            output_path,
            providers=["CPUExecutionProvider"],
        )
        onnx_out = session.run(["output"], {"input": img_input.astype(np.float32)})[0]

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-02, atol=1e-04)
        print("Exported model predictions match original")


def main(config, output_path, disable_flip, test_image_path):
    model = build_model(config)
    model = model.cuda() if torch.cuda.is_available() else model
    export(model, output_path, disable_flip, test_image_path)


def export_model(model_name, pretrained_resource, output_path, disable_flip=False, test_image_path=None, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    main(config, output_path, disable_flip, test_image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to get config for")
    parser.add_argument("--disable_flip", action="store_true", help="Disable flip augmentation")
    parser.add_argument("-o", "--output_path", type=str,
                        required=True, help="Path to output ONNX file")
    parser.add_argument("-t", "--test_image_path", type=str,
                        required=False, help="Path test image that is used for determining whether PyTorch and ONNX results match")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    export_model(args.model, pretrained_resource=args.pretrained_resource, output_path=args.output_path, disable_flip=args.disable_flip, test_image_path=args.test_image_path, **overwrite_kwargs)