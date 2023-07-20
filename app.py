import asyncio
import datetime
import logging
import os
import time
import traceback
import contextlib
from io import StringIO

import edge_tts
import gradio as gr
import librosa
import torch
from fairseq import checkpoint_utils

from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rmvpe import RMVPE
from vc_infer_pipeline import VC

from TTS.api import TTS

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

limitation = os.getenv("SYSTEM") == "spaces"

config = Config()

edge_output_filename = "edge_output.mp3"
edge_tts_voice_list = asyncio.get_event_loop().run_until_complete(
    edge_tts.list_voices()
)
edge_tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in edge_tts_voice_list]

tortoise_output_filename = "tortoise_output.wav"
tortoise_presets = ["ultra_fast", "standard", "high_quality", "single_sample"]
tortoise_voice_root = os.path.join("voices", "tortoise_voices")
tortoise_voices = [
    entry.name for entry in os.scandir(tortoise_voice_root) if entry.is_dir()
]
tortoise_stream = StringIO()
with contextlib.redirect_stdout(tortoise_stream):
    tortoise_tts = TTS("tts_models/en/multi-dataset/tortoise-v2")



rvc_model_root = os.path.join("voices", "rvc_weights")
rvc_models = [
    d
    for d in os.listdir(rvc_model_root)
    if os.path.isdir(os.path.join(rvc_model_root, d))
]
if len(rvc_models) == 0:
    raise ValueError("No model found in `rvc_weights` folder")
rvc_models.sort()
hubert_model = None

print("Loading rmvpe model...")
rmvpe_model = RMVPE("rmvpe.pt", config.is_half, config.device)
print("rmvpe model loaded.")


def rvc_model_data(rvc_model_name):
    # global n_spk, tgt_sr, net_g, vc, cpt, version, index_file
    pth_files = [
        os.path.join(rvc_model_root, rvc_model_name, f)
        for f in os.listdir(os.path.join(rvc_model_root, rvc_model_name))
        if f.endswith(".pth")
    ]
    if len(pth_files) == 0:
        raise ValueError(f"No pth file found in {rvc_model_root}/{rvc_model_name}")
    pth_path = pth_files[0]
    print(f"Loading {pth_path}")
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("RVC Model loaded")
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    # n_spk = cpt["config"][-3]

    index_files = [
        os.path.join(rvc_model_root, rvc_model_name, f)
        for f in os.listdir(os.path.join(rvc_model_root, rvc_model_name))
        if f.endswith(".index")
    ]
    if len(index_files) == 0:
        print("No index file found")
        index_file = ""
    else:
        index_file = index_files[0]
        print(f"Index file found: {index_file}")

    return tgt_sr, net_g, vc, version, index_file, if_f0


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def infer_tortoise_tts(
    rvc_model_name,
    tortoise_tts_text,
    tortoise_tts_voice,
    f0_up_key,
    f0_method,
    index_rate,
    protect,
    tortoise_preset,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
):
    print("------------------")
    print(datetime.datetime.now())
    print("tortoise_tts_text:")
    print(tortoise_tts_text)
    print(f"tortoise_tts_voice: {tortoise_tts_voice}")
    print(f"RVC Model name: {rvc_model_name}")
    print(f"F0: {f0_method}, Key: {f0_up_key}, Index: {index_rate}, Protect: {protect}")

    

    try:
        tgt_sr, net_g, vc, version, index_file, if_f0 = rvc_model_data(rvc_model_name)
        t0 = time.time()
        with contextlib.redirect_stdout(tortoise_stream):
            tortoise_tts.tts_to_file(
                text=tortoise_tts_text,
                file_path=tortoise_output_filename,
                voice_dir=tortoise_voice_root,
                speaker=tortoise_tts_voice,
                preset=tortoise_preset,
            )
        t1 = time.time()
        tortoise_time = t1 - t0
        audio, sr = librosa.load(tortoise_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        
        f0_up_key = int(f0_up_key)

        if not hubert_model:
            load_hubert()
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            edge_output_filename,
            times,
            f0_up_key,
            f0_method,
            index_file,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        info = f"Success. Time: tortoise-tts: {tortoise_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        return (
            info,
            tortoise_output_filename,
            (tgt_sr, audio_opt),
        )
    except EOFError:
        info = (
            "It seems that the tortoise-tts output is not valid. "
            "I have no idea what is going on :)"
        )
        print(info)
        return info, None, None
    except:
        info = traceback.format_exc()
        print(info)
        return info, None, None


def infer_edge_tts(
    rvc_model_name,
    speed,
    edge_tts_text,
    edge_tts_voice,
    f0_up_key,
    f0_method,
    index_rate,
    protect,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
):
    print("------------------")
    print(datetime.datetime.now())
    print("edge_tts_text:")
    print(edge_tts_text)
    print(f"edge_tts_voice: {edge_tts_voice}")
    print(f"RVC Model name: {rvc_model_name}")
    print(f"F0: {f0_method}, Key: {f0_up_key}, Index: {index_rate}, Protect: {protect}")
    try:
        tgt_sr, net_g, vc, version, index_file, if_f0 = rvc_model_data(rvc_model_name)
        t0 = time.time()
        if speed >= 0:
            speed_str = f"+{speed}%"
        else:
            speed_str = f"{speed}%"
        asyncio.run(
            edge_tts.Communicate(
                edge_tts_text, "-".join(edge_tts_voice.split("-")[:-1]), rate=speed_str
            ).save(edge_output_filename)
        )
        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")

        f0_up_key = int(f0_up_key)

        if not hubert_model:
            load_hubert()
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            edge_output_filename,
            times,
            f0_up_key,
            f0_method,
            index_file,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        info = f"Success. Time: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        return (
            info,
            edge_output_filename,
            (tgt_sr, audio_opt),
        )
    except EOFError:
        info = (
            "It seems that the edge-tts output is not valid. "
            "This may occur when the input text and the speaker do not match. "
            "For example, maybe you entered Japanese (without alphabets) text but chose non-Japanese speaker?"
        )
        print(info)
        return info, None, None
    except:
        info = traceback.format_exc()
        print(info)
        return info, None, None


initial_md = """
# RVC text-to-speech webui

This is a text-to-speech webui of RVC models.
"""

app = gr.Blocks()
with app:
    gr.Markdown(initial_md)
    with gr.Row():
        with gr.Column():
            rvc_model_name = gr.Dropdown(
                label="RVC Model", choices=rvc_models, value=rvc_models[0]
            )
            f0_key_up = gr.Number(
                label="Transpose",
                value=0,
            )
        with gr.Column():
            f0_method = gr.Radio(
                label="Pitch extraction method \n(pm: very fast, low quality, rmvpe: a little slow, high quality)",
                choices=["pm", "rmvpe"],  # harvest and crepe is too slow
                value="rmvpe",
                interactive=True,
            )
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label="Index rate",
                value=1,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label="Protect",
                value=0.33,
                step=0.01,
                interactive=True,
            )
    with gr.Row():
        with gr.Tab("Edge-tts"):
            with gr.Column():
                edge_tts_voice = gr.Dropdown(
                    label="Edge-tts speaker (format: language-Country-Name-Gender)",
                    choices=edge_tts_voices,
                    allow_custom_value=False,
                    value="en-US-AriaNeural-Female",
                )
                speed = gr.Slider(
                    minimum=-100,
                    maximum=100,
                    label="Speech speed (%)",
                    value=0,
                    step=10,
                    interactive=True,
                )
                edge_tts_text = gr.Textbox(
                    label="Input Text",
                    value="This is an English text to speech conversation demo.",
                    lines=5,
                    max_lines=40
                )
            with gr.Column():
                edge_button = gr.Button("Convert", variant="primary")
                info_text = gr.Textbox(label="Output info")
        with gr.Tab("Tortoise"):
            with gr.Column():
                tortoise_tts_voice = gr.Dropdown(
                    label="Tortoise voices",
                    choices=tortoise_voices,
                    allow_custom_value=False,
                    value=tortoise_voices[0],
                )
                tortoise_tts_text = gr.Textbox(
                    label="Input Text",
                    value="This is an English text to speech conversation demo.",
                    lines=5,
                    max_lines=40
                )
                tortoise_preset = gr.Dropdown(
                    label="Presets",
                    choices=tortoise_presets,
                    allow_custom_value=False,
                    value=tortoise_presets[0],
                )
            with gr.Column():
                tortoise_button = gr.Button("Convert", variant="primary")
                info_text = gr.Textbox(label="Output info")
        with gr.Column(scale=0):
            tts_output = gr.Audio(label="TTS Voice", type="filepath")
            rvc_tts_output = gr.Audio(label="Result")
        edge_button.click(
            infer_edge_tts,
            [
                rvc_model_name,
                speed,
                edge_tts_text,
                edge_tts_voice,
                f0_key_up,
                f0_method,
                index_rate,
                protect0,
            ],
            [info_text, tts_output, rvc_tts_output],
            api_name="edge_tts",
        )
        tortoise_button.click(
            infer_tortoise_tts,
            [
                rvc_model_name,
                tortoise_tts_text,
                tortoise_tts_voice,
                f0_key_up,
                f0_method,
                index_rate,
                protect0,
                tortoise_preset,
            ],
            [info_text, tts_output, rvc_tts_output],
            api_name="tortoise_tts",
        )
    # with gr.Row():
    #     with gr.Tab('Edge-tts'):
    #         examples = gr.Examples(
    #             examples_per_page=100,
    #             examples=[
    #                 ["これは日本語テキストから音声への変換デモです。", "ja-JP-NanamiNeural-Female"],
    #                 [
    #                     "This is an English text to speech conversation demo.",
    #                     "en-US-AriaNeural-Female",
    #                 ]
    #             ],
    #             inputs=[edge_tts_text, edge_tts_voice],
    #         )
    #     with gr.Tab('tortoise'):
    #         examples = gr.Examples(
    #             examples_per_page=100,
    #             examples=[
    #                 [
    #                     "This is an English text to speech conversation demo.",
    #                     tortoise_voices[0],
    #                 ]
    #             ],
    #             inputs=[tortoise_tts_text, tortoise_tts_voice],
    #         )


app.launch(inbrowser=True)
