import os
import contextlib
import re
import time
from dataclasses import dataclass
from glob import iglob
import torch._inductor.config as cfg
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image
from torch.profiler import ProfilerActivity, profile, record_function
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
from transformers import pipeline

NSFW_THRESHOLD = 0.85

cfg.trace.log_autotuning_results = True # enable the log of autotuning results

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = (
        "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    )
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting seed to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options


@torch.inference_mode()
def main(
    name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    profile_json: str | None = None,
    warmup: bool = True,
    warmup_steps: int | None = 1,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    nsfw_classifier = pipeline(
        "image-classification", model="Falconsai/nsfw_image_detection", device=device
    )

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [
            fn
            for fn in iglob(output_name.format(idx="*"))
            if re.search(r"img_[0-9]+\.jpg$", fn)
        ]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False, backend="inductor")
    model.eval()
    ae = load_ae(name, device="cpu" if offload else torch_device)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    # Optional one-time warm-up to trigger JIT/inductor/triton compilation and memory caches
    if warmup:
        try:
            print("Warm-up: running an unmeasured dry run to compile kernels...")
            # Do not mutate opts.seed here; use a fixed seed for determinism if None
            _seed = opts.seed if opts.seed is not None else 0
            # Prepare input
            xw = get_noise(
                1,
                opts.height,
                opts.width,
                device=torch_device,
                dtype=torch.bfloat16,
                seed=_seed,
            )
            if offload:
                ae = ae.cpu()
                torch.cuda.empty_cache()
                t5, clip = t5.to(torch_device), clip.to(torch_device)
            inpw = prepare(t5, clip, xw, prompt=opts.prompt)
            steps_w = warmup_steps if (warmup_steps is not None and warmup_steps > 0) else 1
            tsteps_w = get_schedule(
                steps_w, inpw["img"].shape[1], shift=(name != "flux-schnell")
            )
            if offload:
                t5, clip = t5.cpu(), clip.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)
            # Denoise and decode
            xw = denoise(model, **inpw, timesteps=tsteps_w, guidance=opts.guidance)
            if offload:
                model.cpu()
                torch.cuda.empty_cache()
                ae.decoder.to(xw.device)
            xw = unpack(xw.float(), opts.height, opts.width)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                xw = ae.decode(xw)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Return AE to CPU to keep device state consistent when offloading
            if offload:
                ae = ae.cpu()
            # Drop warm-up outputs
            del xw
            print("Warm-up: done.")
        except (RuntimeError, ValueError) as e:
            # Warm-up is best-effort; continue even if it fails
            print(f"Warm-up skipped due to error: {e}")

    if loop:
        opts = parse_prompt(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()
        
        activities = [ProfilerActivity.CPU]
        
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        prof_cm = (
            profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True
            )
            if profile_json
            else contextlib.nullcontext()
        )
        
        with prof_cm as prof:
            with record_function("prepare"):
            # prepare input
                x = get_noise(
                    1,
                    opts.height,
                    opts.width,
                    device=torch_device,
                    dtype=torch.bfloat16,
                    seed=opts.seed,
                )
                opts.seed = None
                if offload:
                    ae = ae.cpu()
                    torch.cuda.empty_cache()
                    t5, clip = t5.to(torch_device), clip.to(torch_device)
                inp = prepare(t5, clip, x, prompt=opts.prompt)
                timesteps = get_schedule(
                    opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell")
                )

        # offload TEs to CPU, load model to gpu
            if offload:
                t5, clip = t5.cpu(), clip.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)

            # denoise initial noise
            with record_function("denoise"):
                x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

            # offload model, load autoencoder to gpu
            if offload:
                model.cpu()
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

            # decode latents to pixel space
            with record_function("decode"):
                x = unpack(x.float(), opts.height, opts.width)
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
        if profile_json:
            os.makedirs(os.path.dirname(profile_json) or ".", exist_ok=True)
            trace_path = profile_json
            if not os.path.isabs(trace_path) and not trace_path.startswith(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                trace_path = os.path.join(output_dir, trace_path)
            try:
                if 'prof' in locals() and prof is not None:
                    prof.export_chrome_trace(trace_path)
                    print(f"Wrote Perfetto trace: {trace_path}")
                    print(prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=40
                    ))

            except RuntimeError as e:
                print(f"Failed to export trace to {trace_path}: {e}") 

        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {(t1 - t0)*1000:.1f} ms. Saving {fn}")
        # bring into PIL format and save
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][
            0
        ]

        if nsfw_score < NSFW_THRESHOLD:
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
            exif_data[ExifTags.Base.Make] = "Black Forest Labs"
            exif_data[ExifTags.Base.Model] = name
            if add_sampling_metadata:
                exif_data[ExifTags.Base.ImageDescription] = prompt
            img.save(fn, exif=exif_data, quality=95, subsampling=0)
            idx += 1
        else:
            print("Your generated image may contain NSFW content.")

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
        else:
            opts = None


def app():
    Fire(main)


if __name__ == "__main__":
    app()
