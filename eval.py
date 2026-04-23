import argparse
import os
from functools import partial
import numpy as np
import torch
from torch import amp
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import *
from monai.utils.enums import MetricReduction
from utils.utils import *
from utils.utils import AverageMeter
from monai import data, transforms
from monai.data import *
from models.comer_unetr import ViTCoMerUNETR
from runtime_utils import configure_runtime_warnings, ensure_cuda_available, get_device, resolve_datalist_path, safe_set_resource_limit

configure_runtime_warnings()
safe_set_resource_limit()

parser = argparse.ArgumentParser(description="Segmentation pipeline")
parser.add_argument(
    "--datalist_json", default=None, type=str, help="absolute path to dataset json")
parser.add_argument(
    "--json_list", default="dataset_0.json", type=str, help="dataset json file name relative to data_dir")
parser.add_argument(
    "--data_dir", default="", type=str, help="test_data_path")
parser.add_argument(
    "--save_prediction_path", default="./pred_MY/", type=str, help="test_prediction_path")
parser.add_argument(
    "--trained_pth", default="", type=str, help="trained checkpoint directory")
parser.add_argument("--gpu_id", default=0, type=int, help="single GPU id for inference")
parser.add_argument("--noamp", action="store_true", help="disable autocast during inference")

roi = 96
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=16, type=int, help="number of workers")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--sr_ratio", default=1, type=int, help="multi scale token")

def get_test_loader(args):
    """
    Creates training transforms, constructs a dataset, and returns a dataloader.

    Args:
        args: Command line arguments containing dataset paths and hyperparameters.
    """
    test_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

            # transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    # test_transforms = Compose(
    #     [
    #         LoadImaged(keys=["image", "label"]),
    #         EnsureChannelFirstd(keys=["image", "label"]),
    #         Orientationd(keys=["image", "label"], axcodes="RAS"),
    #         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    #         CropForegroundd(
    #             keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
    #         ),

    #         SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
    #                     mode="constant"),

    #         RandShiftIntensityd(keys="image", offsets=0.1, prob=0),
    #     ]
    # )

    # constructing training dataset


    datalist_json = resolve_datalist_path(args.data_dir, args.json_list, args.datalist_json)
    dataset_list = load_decathlon_datalist(datalist_json, True, "validation", base_dir=args.data_dir)


    print('test len {}'.format(len(dataset_list)))

    test_ds = Dataset(data=dataset_list, transform=test_transforms)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=None, pin_memory=True
    )
    return test_loader, test_transforms


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    ensure_cuda_available()
    device = get_device(args.gpu_id)
    os.makedirs(args.save_prediction_path, exist_ok=True)

    test_loader, test_transforms = get_test_loader(args)
    # model=MiT(num_classes=args.out_channels)
    # model = SwinUNETR(
    #     # img_size=(args.roi_x, args.roi_y, args.roi_z),
    #     in_channels=args.in_channels,
    #     out_channels=args.out_channels,
    #     feature_size=48,
    #     drop_rate=0.0,
    #     attn_drop_rate=0.0,
    #     dropout_path_rate=0.0,
    #     use_checkpoint=args.use_checkpoint,
    #     use_v2=True
    # )
    model=ViTCoMerUNETR( img_size=(96, 96, 96),
    in_channels=args.in_channels,
    out_channels=args.out_channels,
    feature_size=args.feature_size,
    )
    # model= ConvViT3dUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
    #                            in_channels=args.in_channels,
    #                            out_channels=args.out_channels,
    #                            feature_size=args.feature_size,
    #                            args=args,
    #                            )
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    checkpoint = torch.load(args.trained_pth, map_location="cpu", weights_only=False)
    model_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    model.to(device)

    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    post_transforms = Compose([EnsureTyped(keys=["pred"]),
                               Invertd(keys=["pred"],
                                       transform=test_transforms,
                                       orig_keys="image",
                                       meta_keys="pred_meta_dict",
                                       orig_meta_keys="image_meta_dict",
                                       meta_key_postfix="meta_dict",
                                       nearest_interp=True,
                                       to_tensor=True),
                               AsDiscreted(keys="pred", argmax=False, to_onehot=None),
                               SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.save_prediction_path,
                                          separate_folder=False, folder_layout=None,
                                          resample=True),
                               ])

    acc_func = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    run_acc = AverageMeter()
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            torch.cuda.empty_cache()

            data = batch_data["image"]
            data = data.to(device, non_blocking=True)

            label = batch_data["label"]
            label = label.to(device, non_blocking=True)

            # name = batch_data['name'][0]

            with amp.autocast("cuda", enabled=args.amp):
                logits = model_inferer(data)

            val_labels_list = decollate_batch(label)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            print(np.mean(run_acc.avg))
            output = logits.argmax(1)
            batch_data['pred'] = output.unsqueeze(1)
            batch_data = [post_transforms(i) for i in
                          decollate_batch(batch_data)]

            # os.rename(os.path.join(args.save_prediction_path, name[:-7]+'_trans.nii.gz'),
            #           os.path.join(args.save_prediction_path, name[:-12]+'.nii.gz'))


if __name__ == "__main__":
    main()
