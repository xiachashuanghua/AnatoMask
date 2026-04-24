from __future__ import annotations

import base64
import inspect
import json
import mimetypes

import gradio as gr

from launcher.job_manager import (
    cancel_job,
    get_job_details,
    list_jobs,
    list_train_checkpoints,
    start_infer_job,
    start_train_job,
)
from runtime_utils import PROJECT_ROOT, dumps_pretty


DEFAULT_PRETRAIN = PROJECT_ROOT / "model_step60000.pt"
LOGO_PATH = PROJECT_ROOT / "logo.png"
FAVICON_PATH = PROJECT_ROOT / "logo_no_cha.png"
UI_STATE_PATH = PROJECT_ROOT / "webui_runs" / "ui_state.json"
TRAIN_FORM_FIELDS = [
    "gpu_id",
    "data_dir",
    "json_list",
    "cache_dir",
    "pretrained_root",
    "checkpoint",
    "feature_size",
    "in_channels",
    "out_channels",
    "max_epochs",
    "batch_size",
    "sw_batch_size",
    "optim_lr",
    "val_every",
    "workers",
    "use_persistent_dataset",
    "rand_flipd_prob",
    "rand_rotate90d_prob",
    "rand_scale_intensityd_prob",
    "rand_shift_intensityd_prob",
    "infer_overlap",
    "roi_x",
    "roi_y",
    "roi_z",
    "space_x",
    "space_y",
    "space_z",
    "a_min",
    "a_max",
    "b_min",
    "b_max",
    "noamp",
]
INFER_FORM_FIELDS = [
    "gpu_id",
    "data_dir",
    "json_list",
    "trained_pth",
    "save_prediction_path",
    "feature_size",
    "in_channels",
    "out_channels",
    "sw_batch_size",
    "workers",
    "infer_overlap",
    "roi_x",
    "roi_y",
    "roi_z",
    "space_x",
    "space_y",
    "space_z",
    "a_min",
    "a_max",
    "b_min",
    "b_max",
    "noamp",
]


def _default_pretrained() -> str:
    return str(DEFAULT_PRETRAIN) if DEFAULT_PRETRAIN.exists() else ""


def _default_ui_state() -> dict:
    return {
        "train": {
            "gpu_id": 0,
            "feature_size": 48,
            "out_channels": 14,
            "in_channels": 1,
            "data_dir": "",
            "json_list": "dataset_0.json",
            "cache_dir": "",
            "pretrained_root": _default_pretrained(),
            "checkpoint": "",
            "max_epochs": 300,
            "batch_size": 1,
            "sw_batch_size": 4,
            "workers": 4,
            "use_persistent_dataset": True,
            "rand_flipd_prob": 0.2,
            "rand_rotate90d_prob": 0.2,
            "rand_scale_intensityd_prob": 0.1,
            "rand_shift_intensityd_prob": 0.5,
            "optim_lr": 3e-4,
            "val_every": 10,
            "infer_overlap": 0.75,
            "noamp": False,
            "roi_x": 96,
            "roi_y": 96,
            "roi_z": 96,
            "space_x": 1.5,
            "space_y": 1.5,
            "space_z": 2.0,
            "a_min": -175.0,
            "a_max": 250.0,
            "b_min": 0.0,
            "b_max": 1.0,
        },
        "infer": {
            "gpu_id": 0,
            "feature_size": 48,
            "out_channels": 14,
            "in_channels": 1,
            "data_dir": "",
            "json_list": "dataset_0.json",
            "trained_pth": "",
            "save_prediction_path": "",
            "sw_batch_size": 4,
            "workers": 4,
            "infer_overlap": 0.75,
            "noamp": False,
            "roi_x": 96,
            "roi_y": 96,
            "roi_z": 96,
            "space_x": 1.5,
            "space_y": 1.5,
            "space_z": 2.0,
            "a_min": -175.0,
            "a_max": 250.0,
            "b_min": 0.0,
            "b_max": 1.0,
        },
    }


def _load_ui_state() -> dict:
    state = _default_ui_state()
    if not UI_STATE_PATH.exists():
        return state
    try:
        saved = json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return state
    for section in ("train", "infer"):
        if isinstance(saved.get(section), dict):
            state[section].update(saved[section])
    return state


def _save_ui_state(section: str, payload: dict) -> None:
    UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = _load_ui_state()
    state[section].update(payload)
    UI_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _payload_from_fields(field_names: list[str], values: tuple) -> dict:
    return {name: value for name, value in zip(field_names, values)}


def _header_html() -> str:
    logo_html = ""
    if LOGO_PATH.exists():
        mime_type = mimetypes.guess_type(LOGO_PATH.name)[0] or "image/png"
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        logo_html = (
            f'<img src="data:{mime_type};base64,{encoded}" '
            'alt="AnatoMask logo" '
            'style="display:block; width:min(240px, 40vw); height:auto; object-fit:contain;" />'
        )

    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; gap:10px; padding:12px 0 4px 0;">
      {logo_html}
      <h1 style="text-align:center; margin:0;">AnatoMask</h1>
    </div>
    """


def _persist_train_form(*values):
    _save_ui_state("train", _payload_from_fields(TRAIN_FORM_FIELDS, values))


def _persist_infer_form(*values):
    _save_ui_state("infer", _payload_from_fields(INFER_FORM_FIELDS, values))


def _checkpoint_choices():
    choices = list_train_checkpoints()
    infer_state = _load_ui_state().get("infer", {})
    saved_value = infer_state.get("trained_pth")
    valid_values = {value for _, value in choices}
    value = saved_value if saved_value in valid_values else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def _train_submit(
    gpu_id,
    data_dir,
    json_list,
    cache_dir,
    pretrained_root,
    checkpoint,
    feature_size,
    in_channels,
    out_channels,
    max_epochs,
    batch_size,
    sw_batch_size,
    optim_lr,
    val_every,
    workers,
    use_persistent_dataset,
    rand_flipd_prob,
    rand_rotate90d_prob,
    rand_scale_intensityd_prob,
    rand_shift_intensityd_prob,
    infer_overlap,
    roi_x,
    roi_y,
    roi_z,
    space_x,
    space_y,
    space_z,
    a_min,
    a_max,
    b_min,
    b_max,
    noamp,
):
    config = _payload_from_fields(
        TRAIN_FORM_FIELDS,
        (
            gpu_id,
            data_dir,
            json_list,
            cache_dir,
            pretrained_root,
            checkpoint,
            feature_size,
            in_channels,
            out_channels,
            max_epochs,
            batch_size,
            sw_batch_size,
            optim_lr,
            val_every,
            workers,
            use_persistent_dataset,
            rand_flipd_prob,
            rand_rotate90d_prob,
            rand_scale_intensityd_prob,
            rand_shift_intensityd_prob,
            infer_overlap,
            roi_x,
            roi_y,
            roi_z,
            space_x,
            space_y,
            space_z,
            a_min,
            a_max,
            b_min,
            b_max,
            noamp,
        ),
    )
    _save_ui_state("train", config)
    return start_train_job(config)


def _infer_submit(
    gpu_id,
    data_dir,
    json_list,
    trained_pth,
    save_prediction_path,
    feature_size,
    in_channels,
    out_channels,
    sw_batch_size,
    workers,
    infer_overlap,
    roi_x,
    roi_y,
    roi_z,
    space_x,
    space_y,
    space_z,
    a_min,
    a_max,
    b_min,
    b_max,
    noamp,
):
    config = _payload_from_fields(
        INFER_FORM_FIELDS,
        (
            gpu_id,
            data_dir,
            json_list,
            trained_pth,
            save_prediction_path,
            feature_size,
            in_channels,
            out_channels,
            sw_batch_size,
            workers,
            infer_overlap,
            roi_x,
            roi_y,
            roi_z,
            space_x,
            space_y,
            space_z,
            a_min,
            a_max,
            b_min,
            b_max,
            noamp,
        ),
    )
    _save_ui_state("infer", config)
    return start_infer_job(config)


def _job_choices():
    jobs = list_jobs()
    return gr.update(choices=jobs, value=jobs[0] if jobs else None)


def _job_details(job_id):
    state, config, log_tail, artifacts = get_job_details(job_id)
    return state, config, log_tail, artifacts


def _cancel(job_id):
    state = cancel_job(job_id)
    return state, dumps_pretty(state) if state else "{}"


def build_app() -> gr.Blocks:
    ui_state = _load_ui_state()
    train_state = ui_state["train"]
    infer_state = ui_state["infer"]
    checkpoint_choices = list_train_checkpoints()
    checkpoint_values = {value for _, value in checkpoint_choices}
    infer_checkpoint_value = (
        infer_state.get("trained_pth")
        if infer_state.get("trained_pth") in checkpoint_values
        else (checkpoint_choices[0][1] if checkpoint_choices else None)
    )

    with gr.Blocks(title="AnatoMask") as demo:
        gr.HTML(_header_html())

        with gr.Tab("Train"):
            with gr.Row():
                train_gpu_id = gr.Number(value=train_state["gpu_id"], precision=0, label="GPU ID")
                train_feature_size = gr.Number(value=train_state["feature_size"], precision=0, label="feature_size")
                train_out_channels = gr.Number(value=train_state["out_channels"], precision=0, label="out_channels")
                train_in_channels = gr.Number(value=train_state["in_channels"], precision=0, label="in_channels")
            train_data_dir = gr.Textbox(value=train_state["data_dir"], label="Dataset directory `data_dir`")
            train_json_list = gr.Textbox(value=train_state["json_list"], label="Dataset JSON filename `json_list`")
            gr.Markdown("Note: `cache_dir` is only used when `Use PersistentDataset` is enabled.")
            train_cache_dir = gr.Textbox(value=train_state["cache_dir"], label="Cache directory `cache_dir` (auto-created if empty)")
            train_pretrained_root = gr.Textbox(value=train_state["pretrained_root"], label="Pretrained weights `pretrained_root` (optional)")
            train_checkpoint = gr.Textbox(value=train_state["checkpoint"], label="Resume checkpoint `checkpoint` (optional)")
            with gr.Row():
                train_max_epochs = gr.Number(value=train_state["max_epochs"], precision=0, label="max_epochs")
                train_batch_size = gr.Number(value=train_state["batch_size"], precision=0, label="batch_size")
                train_sw_batch_size = gr.Number(value=train_state["sw_batch_size"], precision=0, label="sw_batch_size")
                train_workers = gr.Number(value=train_state["workers"], precision=0, label="workers")
            with gr.Row():
                train_optim_lr = gr.Number(value=train_state["optim_lr"], label="optim_lr")
                train_val_every = gr.Number(value=train_state["val_every"], precision=0, label="val_every")
                train_infer_overlap = gr.Number(value=train_state["infer_overlap"], label="infer_overlap")
                train_noamp = gr.Checkbox(value=train_state["noamp"], label="Disable AMP")
            with gr.Accordion("Advanced preprocessing", open=False):
                with gr.Row():
                    train_use_persistent_dataset = gr.Checkbox(
                        value=train_state["use_persistent_dataset"],
                        label="Use PersistentDataset",
                    )
                    train_rand_flipd_prob = gr.Number(value=train_state["rand_flipd_prob"], label="RandFlipd_prob")
                    train_rand_rotate90d_prob = gr.Number(
                        value=train_state["rand_rotate90d_prob"],
                        label="RandRotate90d_prob",
                    )
                with gr.Row():
                    train_rand_scale_intensityd_prob = gr.Number(
                        value=train_state["rand_scale_intensityd_prob"],
                        label="RandScaleIntensityd_prob",
                    )
                    train_rand_shift_intensityd_prob = gr.Number(
                        value=train_state["rand_shift_intensityd_prob"],
                        label="RandShiftIntensityd_prob",
                    )
                with gr.Row():
                    train_roi_x = gr.Number(value=train_state["roi_x"], precision=0, label="roi_x")
                    train_roi_y = gr.Number(value=train_state["roi_y"], precision=0, label="roi_y")
                    train_roi_z = gr.Number(value=train_state["roi_z"], precision=0, label="roi_z")
                with gr.Row():
                    train_space_x = gr.Number(value=train_state["space_x"], label="space_x")
                    train_space_y = gr.Number(value=train_state["space_y"], label="space_y")
                    train_space_z = gr.Number(value=train_state["space_z"], label="space_z")
                with gr.Row():
                    train_a_min = gr.Number(value=train_state["a_min"], label="a_min")
                    train_a_max = gr.Number(value=train_state["a_max"], label="a_max")
                    train_b_min = gr.Number(value=train_state["b_min"], label="b_min")
                    train_b_max = gr.Number(value=train_state["b_max"], label="b_max")
            train_submit = gr.Button("Start training", variant="primary")
            train_inputs = [
                train_gpu_id,
                train_data_dir,
                train_json_list,
                train_cache_dir,
                train_pretrained_root,
                train_checkpoint,
                train_feature_size,
                train_in_channels,
                train_out_channels,
                train_max_epochs,
                train_batch_size,
                train_sw_batch_size,
                train_optim_lr,
                train_val_every,
                train_workers,
                train_use_persistent_dataset,
                train_rand_flipd_prob,
                train_rand_rotate90d_prob,
                train_rand_scale_intensityd_prob,
                train_rand_shift_intensityd_prob,
                train_infer_overlap,
                train_roi_x,
                train_roi_y,
                train_roi_z,
                train_space_x,
                train_space_y,
                train_space_z,
                train_a_min,
                train_a_max,
                train_b_min,
                train_b_max,
                train_noamp,
            ]
            with gr.Row():
                train_job_id = gr.Textbox(label="Job ID")
                train_job_dir = gr.Textbox(label="Job directory")
            with gr.Row():
                train_status = gr.Textbox(label="Status")
                train_command = gr.Textbox(label="Launch command")
            train_submit.click(
                fn=_train_submit,
                inputs=train_inputs,
                outputs=[train_job_id, train_job_dir, train_status, train_command],
            )
            for component in train_inputs:
                component.change(fn=_persist_train_form, inputs=train_inputs, outputs=[])

        with gr.Tab("Inference"):
            with gr.Row():
                infer_gpu_id = gr.Number(value=infer_state["gpu_id"], precision=0, label="GPU ID")
                infer_feature_size = gr.Number(value=infer_state["feature_size"], precision=0, label="feature_size")
                infer_out_channels = gr.Number(value=infer_state["out_channels"], precision=0, label="out_channels")
                infer_in_channels = gr.Number(value=infer_state["in_channels"], precision=0, label="in_channels")
            infer_data_dir = gr.Textbox(value=infer_state["data_dir"], label="Dataset directory `data_dir`")
            infer_json_list = gr.Textbox(value=infer_state["json_list"], label="Dataset JSON filename `json_list`")
            with gr.Row():
                refresh_checkpoints = gr.Button("Refresh trained models")
                infer_trained_pth = gr.Dropdown(
                    choices=checkpoint_choices,
                    value=infer_checkpoint_value,
                    label="Trained model checkpoint",
                    allow_custom_value=False,
                )
            infer_save_prediction_path = gr.Textbox(value=infer_state["save_prediction_path"], label="Output directory `save_prediction_path` (auto-created if empty)")
            with gr.Row():
                infer_sw_batch_size = gr.Number(value=infer_state["sw_batch_size"], precision=0, label="sw_batch_size")
                infer_workers = gr.Number(value=infer_state["workers"], precision=0, label="workers")
                infer_infer_overlap = gr.Number(value=infer_state["infer_overlap"], label="infer_overlap")
                infer_noamp = gr.Checkbox(value=infer_state["noamp"], label="Disable AMP")
            with gr.Accordion("Advanced preprocessing", open=False):
                with gr.Row():
                    infer_roi_x = gr.Number(value=infer_state["roi_x"], precision=0, label="roi_x")
                    infer_roi_y = gr.Number(value=infer_state["roi_y"], precision=0, label="roi_y")
                    infer_roi_z = gr.Number(value=infer_state["roi_z"], precision=0, label="roi_z")
                with gr.Row():
                    infer_space_x = gr.Number(value=infer_state["space_x"], label="space_x")
                    infer_space_y = gr.Number(value=infer_state["space_y"], label="space_y")
                    infer_space_z = gr.Number(value=infer_state["space_z"], label="space_z")
                with gr.Row():
                    infer_a_min = gr.Number(value=infer_state["a_min"], label="a_min")
                    infer_a_max = gr.Number(value=infer_state["a_max"], label="a_max")
                    infer_b_min = gr.Number(value=infer_state["b_min"], label="b_min")
                    infer_b_max = gr.Number(value=infer_state["b_max"], label="b_max")
            infer_submit = gr.Button("Start inference", variant="primary")
            infer_inputs = [
                infer_gpu_id,
                infer_data_dir,
                infer_json_list,
                infer_trained_pth,
                infer_save_prediction_path,
                infer_feature_size,
                infer_in_channels,
                infer_out_channels,
                infer_sw_batch_size,
                infer_workers,
                infer_infer_overlap,
                infer_roi_x,
                infer_roi_y,
                infer_roi_z,
                infer_space_x,
                infer_space_y,
                infer_space_z,
                infer_a_min,
                infer_a_max,
                infer_b_min,
                infer_b_max,
                infer_noamp,
            ]
            with gr.Row():
                infer_job_id = gr.Textbox(label="Job ID")
                infer_job_dir = gr.Textbox(label="Job directory")
            with gr.Row():
                infer_status = gr.Textbox(label="Status")
                infer_command = gr.Textbox(label="Launch command")
            infer_submit.click(
                fn=_infer_submit,
                inputs=infer_inputs,
                outputs=[infer_job_id, infer_job_dir, infer_status, infer_command],
            )
            refresh_checkpoints.click(fn=_checkpoint_choices, outputs=[infer_trained_pth])
            for component in infer_inputs:
                component.change(fn=_persist_infer_form, inputs=infer_inputs, outputs=[])

        with gr.Tab("Jobs"):
            refresh_jobs = gr.Button("Refresh job list")
            job_selector = gr.Dropdown(choices=list_jobs(), label="Select job", allow_custom_value=True)
            inspect = gr.Button("Inspect")
            cancel = gr.Button("Cancel job")
            job_state = gr.JSON(label="state.json")
            job_config = gr.JSON(label="config.json")
            job_log = gr.Textbox(label="Log tail", lines=20)
            job_artifacts = gr.Textbox(label="Artifacts", lines=10)
            cancel_result = gr.Textbox(label="Cancel result")

            refresh_jobs.click(fn=_job_choices, outputs=[job_selector])
            inspect.click(fn=_job_details, inputs=[job_selector], outputs=[job_state, job_config, job_log, job_artifacts])
            cancel.click(fn=_cancel, inputs=[job_selector], outputs=[job_state, cancel_result])

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860) -> None:
    app = build_app()
    launch_kwargs = {
        "server_name": host,
        "server_port": port,
    }
    if FAVICON_PATH.exists():
        launch_kwargs["favicon_path"] = str(FAVICON_PATH)
    try:
        signature = inspect.signature(app.launch)
        launch_kwargs = {
            name: value
            for name, value in launch_kwargs.items()
            if name in signature.parameters
        }
    except (TypeError, ValueError):
        pass
    app.launch(**launch_kwargs)
