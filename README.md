# AnatoMask

<p align="center">
  <img src="logo.png" alt="AnatoMask logo" width="260">
</p>

`AnatoMask` is a local 3D medical image segmentation tool that supports:

- starting model training from a graphical interface
- starting model inference from a graphical interface
- automatically saving every training and inference job
- automatically remembering the last parameters you entered
- automatically listing trained models on the inference page

## 1. How to Start

### Windows

For most users, the simplest way is to double-click:

```bat
start_anatomask.bat
```

If you prefer PowerShell, you can also run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_anatomask.ps1
```

### Linux

```bash
sh start_anatomask.sh
```

### Desktop Client

If you do not want to use a browser, you can launch the desktop client directly.

Windows:

```bat
client\start_anatomask_client.bat
```

Linux:

```bash
sh client/start_anatomask_client.sh
```

The desktop client and the web UI use:

- the same Python environment
- the same training and inference jobs
- the same saved parameters

## 2. What You Will See After Launch

If you start the web UI, open:

```text
http://127.0.0.1:7860
```

The interface contains 3 main tabs:

1. `Train`
2. `Inference`
3. `Jobs`

## 3. How to Organize Training Data

The training data directory follows the same organization style as `nnUNet`.

A recommended training dataset structure is:

```text
dataset_train/
|-- imagesTr/
|-- labelsTr/
`-- dataset.json
```

Where:

- `imagesTr/` stores the training images
- `labelsTr/` stores the corresponding labels
- `dataset.json` describes the training and validation cases

Minimal JSON example for training:

```json
{
  "training": [
    {
      "image": "imagesTr/case001_0000.nii.gz",
      "label": "labelsTr/case001.nii.gz"
    }
  ],
  "validation": [
    {
      "image": "imagesTr/case001_0000.nii.gz",
      "label": "labelsTr/case001.nii.gz"
    }
  ]
}
```

Notes for training:

- `data_dir` should be the directory that contains `imagesTr`, `labelsTr`, and the JSON file
- `json_list` should be only the JSON filename, for example `train.json`
- paths inside the JSON file are written relative to `data_dir`
- image filenames should follow the `nnUNet` naming style, for example `case001_0000.nii.gz`
- label filenames should use the same case ID, for example `case001.nii.gz`
- training requires `label`

## 4. How to Organize Inference Data

For clinical use, inference only needs images and does not require labels.

A recommended inference dataset structure is:

```text
dataset_infer/
|-- imagesTs/
`-- infer.json
```

Where:

- `imagesTs/` stores the images to be segmented
- `infer.json` describes the cases for inference

Minimal JSON example for inference:

```json
{
  "validation": [
    {
      "image": "imagesTs/case001_0000.nii.gz"
    }
  ]
}
```

This shorter format is also supported:

```json
{
  "test": [
    "imagesTs/case001_0000.nii.gz",
    "imagesTs/case002_0000.nii.gz"
  ]
}
```

Notes for inference:

- `data_dir` should be the directory that contains the image folder and the JSON file
- `json_list` should be only the JSON filename, for example `infer.json`
- paths inside the JSON file are written relative to `data_dir`
- inference does not require `label`
- if a `label` field is still present in the inference JSON, it will be ignored

## 5. How to Train

Open the `Train` tab.

### Required Fields in Most Cases

- `GPU ID`
  Usually `0`
- `Dataset directory data_dir`
  The path to your dataset folder
- `Dataset JSON filename json_list`
  For example `train.json`

### Common Parameters

- `feature_size`
  For a first run, keeping the default is usually fine
- `in_channels`
  For single-channel CT or MR, this is usually `1`
- `out_channels`
  This should match the number of segmentation classes
  For example, if your label IDs are `0-13`, set this to `14`
- `max_epochs`
  Number of training epochs
- `batch_size`
  If GPU memory is limited, start with a small value such as `1`
- `workers`
  Number of data loading worker processes

### Optional Parameters

- `Pretrained weights pretrained_root`
  Optional
  If provided, training starts from pretrained weights
- `Resume checkpoint checkpoint`
  Optional
  Only needed if you want to continue a previous interrupted training job
- `Cache directory cache_dir`
  Only takes effect when `Use PersistentDataset` is enabled
  If left empty, it will be created automatically

### Advanced Preprocessing Parameters

Inside the `Advanced preprocessing` panel you can configure:

- `Use PersistentDataset`
- data augmentation probabilities
- voxel spacing: `space_x / space_y / space_z`
- input patch size: `roi_x / roi_y / roi_z`
- intensity range: `a_min / a_max / b_min / b_max`

If you are not sure what these parameters mean, it is recommended to keep the default values for the first run.

### Start Training

Click:

```text
Start training
```

After you click it, the interface will return:

- `Job ID`
- `Job directory`
- `Status`
- `Launch command`

Training runs in the background, so the interface does not freeze.

## 6. How to Run Inference

Open the `Inference` tab.

### Fill in These Fields First

- `GPU ID`
  Usually `0`
- `Dataset directory data_dir`
- `Dataset JSON filename json_list`

### Choose a Trained Model

First click:

```text
Refresh trained models
```

Then choose a model in:

```text
Trained model checkpoint
```

This dropdown automatically collects model files generated by previous training jobs, including:

- `model.pt`
- `model_final.pt`

### Output Directory

- `Output directory save_prediction_path`
  Optional
  If left empty, the program will automatically create an output directory for this inference job

### Start Inference

Click:

```text
Start inference
```

The interface will also return:

- `Job ID`
- `Job directory`
- `Status`
- `Launch command`

## 7. How to Check Logs and Results

Open the `Jobs` tab.

There you can:

- refresh the job list
- inspect the status of a specific job
- view `state.json`
- view `config.json`
- view the latest log output
- view files generated by the job
- cancel a running job

If training or inference fails, this is the first place you should check.

## 8. Where Files Are Saved

### Saved UI Parameters

The parameters you enter in the interface are automatically saved to:

```text
webui_runs/ui_state.json
```

After restarting, most of your previous inputs will still be there.

### Training Job Outputs

Each training run creates its own directory:

```text
webui_runs/jobs/train-YYYYMMDD-HHMMSS-xxxxxxxx/
```

Common files include:

- `stdout.log`
- `state.json`
- `config.json`
- `command.txt`
- `artifacts/model.pt`
- `artifacts/model_final.pt`

### Inference Job Outputs

Each inference run also creates its own directory:

```text
webui_runs/jobs/infer-YYYYMMDD-HHMMSS-xxxxxxxx/
```

If you do not manually set an output directory, predictions are usually saved to:

```text
webui_runs/jobs/infer-.../predictions/
```

### Cache Directory

If `Use PersistentDataset` is enabled and `cache_dir` is left empty, the cache will be created automatically at:

```text
webui_runs/cache/<job-id>/
```
