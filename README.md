# underfit

**LoRA finetuning for [Stable Audio 3](https://huggingface.co/stabilityai/stable-audio-3-medium).** Clone, install, click **+ Finetune**, and a small `.safetensors` adapter lands in your dashboard. Plug it into any SA3 inference setup (the bundled Gradio, ComfyUI nodes, your own scripts) and the model now writes in your dataset's style.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dada-bots/underfit/blob/main/underfit-colab.ipynb)  ![python](https://img.shields.io/badge/python-3.10-blue)  ![license](https://img.shields.io/badge/license-MIT-green)

> Throughout this README **LoRA** is used as a catch-all term for the whole adapter family — DoRA, BoRA, the -XS variants, and original LoRA included. See the [Adapter formats](#adapter-formats) table for the full list.

---

## Quickstart — local GPU box

Linux box with an NVIDIA GPU, `git`, and `curl`. Everything else gets fetched automatically.

```bash
git clone https://github.com/dada-bots/underfit && cd underfit
./install.sh                            # ~5 min: installs uv, syncs deps, clones SA3, downloads model packs
./run.sh                                # serves the dashboard on http://localhost:8787
```

Open the URL in a browser, click **+ Finetune**, follow the form.

> **First time only** — open https://huggingface.co/stabilityai/stable-audio-3-medium and click *Agree and access repository*. The three SA3 ARC repos share one license; one click unlocks all three. Approval is instant. Without it the install fails with `401 Unauthorized` on the ARC checkpoint downloads. (The base checkpoints aren't gated.)

### Install options

```bash
./install.sh --backend sat              # opt into stable-audio-tools (default is sa3)
./install.sh --no-setup                 # stop after `uv sync`, skip the model-download wizard
./run.sh                                # warns if you haven't run install.sh yet
```

If you want a specific subset of model packs without the interactive wizard:

```bash
.venv/bin/python -m underfit.cli.setup --models sa3-sm-music,sa3-sm-sfx
```

---

## Requirements

|                | |
|---             |---|
| **OS**         | Linux. macOS / Windows untested. |
| **GPU**        | NVIDIA. ≥16 GB VRAM ideal. 12 GB works with SA3-small + small latent crop + batch 1. |
| **Python**     | 3.10 (auto-fetched by `uv`). |
| **Disk**       | Plan for ~17 GB per SA3-medium pack, ~7 GB per small pack. All three = ~31 GB of checkpoints. Datasets add a few hundred MB each. |
| **HF account** | Free. Needed to accept the [SA3 license](https://huggingface.co/stabilityai/stable-audio-3-medium) once. |

| GPU tier   | Fit | Notes |
|---         |---|---|
| H100 (80GB)| ✅ | Best. Full batch, full latent length, demos in seconds. |
| A100 / L4  | ✅ | Comfortable. |
| G4 / 4090  | ✅ | Fine for SA3-medium at sensible settings. |
| T4 (16GB)  | ⚠️ | Slow but works on SA3-small + latent-length ≤128 + batch_size 1. |
| CPU only   | ❌ | Won't train. |

---

## What you can do

- **Encode any folder of audio** into pre-encoded latents — multi-GPU sharded if you have more than one card. Supported: WAV / FLAC / MP3 / OGG / OPUS / M4A / AIFF.
- **Train a LoRA** with live loss curves and a "loss-by-timestep" chart that splits the loss across the diffusion sigma range. Demo MP3s + tinted-spectrogram previews get generated every N steps.
- **Launch a Gradio inference UI per checkpoint** with a 🚀 button. Dial LoRA strength <1.0 to blend with the base model, skip the LoRA on early denoising steps for cleaner song structure, do audio2audio style transfer, or inpaint regions of an input.
- **Blend multiple LoRAs at once** via `run_gradio.py --lora-ckpt-path lora_a.safetensors lora_b.safetensors`.
- **Continue training from a previous LoRA** — upload a `.safetensors` in *New Finetune* and the run picks up where it left off. Lets you mix datasets / styles iteratively.
- **Auto-resume across disconnects** — runs are detached subprocesses managed by the dashboard's `RunsRegistry`; the dashboard re-reads `runs.json` on launch so closing the tab or restarting the server doesn't kill your training.

---

## Adapter formats

| Format        | What it adds | When |
|---            |---|---|
| **LoRA**      | `lora_A`, `lora_B` | The original low-rank adapter. Fast, well-understood. |
| **DoRA**      | LoRA + a per-column magnitude vector | **Recommended default.** Generally better-quality fits than vanilla LoRA. |
| **BoRA**      | LoRA + per-row *and* per-column magnitudes | Like DoRA but scales in both dimensions. |
| **LoRA-XS / DoRA-XS / BoRA-XS** | Same as above, but the rank-`r` matrices are factored against fixed SVD bases instead of being free parameters | "Extra-small" variants. Smaller files, slightly less capacity. |

Pick one in the **LoRA type** dropdown of *New Finetune*. All variants produce a single `.safetensors` file you can load anywhere.

---

## Models supported

| Pack | HF repo | Gated? | Best for |
|---|---|---|---|
| `sa3-medium`   | [`stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium)             | ARC: yes | General music + SFX. Big quality. ~17 GB. |
| `sa3-sm-music` | [`stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music)   | ARC: yes | Music-focused, small. ~7 GB. T4-friendly. |
| `sa3-sm-sfx`   | [`stable-audio-3-small-sfx`](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)       | ARC: yes | SFX-focused, small. ~7 GB. T4-friendly. |

Each pack ships in two parts:
- **Base** (rectified-flow) — used for LoRA finetuning + as the inference teacher.
- **ARC** (adversarial-consistency, fast inference) — distilled, runs in 4–8 steps instead of 50.

The three ARC repos share one license — agree on any one of them and all three unlock at once.

---

## Step-by-step: your first LoRA

### 1. Get your audio in a folder

10+ minutes of audio is the floor; 30+ min is better. **Quality > quantity.** One coherent style per dataset (one artist, one genre, one SFX category). Mixed bags train into mush. The dashboard lets you tick/untick individual files after the scan, so you don't have to pre-curate.

### 2. (Optional) Add metadata for prompts

Each clip can have key-value metadata that the LoRA learns to associate with the audio. underfit looks in this order:

**a. JSON sidecar** (winner — overrides any embedded tags). A file with the same name as the audio, `.json` extension, sitting either:

- right next to the audio: `my-songs/01.wav` + `my-songs/01.json`, or
- in a sibling `json/` folder: `my-songs/01.wav` + `my-songs/json/01.json`

Any string/number values are kept — both standard keys and custom ones. Useful for per-clip captions, BPMs scraped from somewhere, prompt overrides, anything:

```json
{
  "title": "Eclipse",
  "genre": "deathcore",
  "bpm": 145,
  "mood": "melancholic",
  "prompt": "slow blackened deathcore with clean choir vocals"
}
```

**b. Embedded audio tags** (ID3 on MP3, Vorbis on FLAC/OGG, M4A atoms, etc.) — read via the `audio_metadata` library. Recognised keys: `title`, `artist`, `album`, `genre`, `label`, `date`, `composer`, `bpm`. If your music library is already tagged in Picard / Mp3tag / iTunes, you get those for free.

**c. Skip metadata entirely** — totally fine. In *Configure prompts* you can compose prompts from the file path (e.g. `/sfx/explosions/fireworks/03.wav` becomes the prompt — folder structure becomes meaning), or a fixed string for every clip, or both mixed.

### 3. Create a dataset

In the dashboard click **+ Dataset**. Paste the path to your audio folder. The dashboard scans, lets you tick/untick files, then spawns a pre-encoding subprocess on the GPU. Each audio file becomes a `.npy` (encoded latent) + `.json` (metadata) pair. When the dataset appears in the **Datasets** panel it's ready to train against.

> On a T4 this can take a while. The encoding panel updates as each file finishes.

### 4. New Finetune — fill in the form

In the dashboard click **+ Finetune**.

| Field | What to put | Why |
|---|---|---|
| **Name** | `my-first-lora` (alphanumeric + hyphens) | Run ID + `.safetensors` filename |
| **Model** | `sa3-medium` | Base model to finetune against |
| **Dataset** | the one you created in Step 3 | Pre-encoded latents |
| **LoRA type** | **`DoRA`** | Recommended. `DoRA-XS` for smaller files. |
| **LoRA rank** | `16` | Capacity. Higher = more parameters = heavier file + more overfitting risk. |
| **Steps** | `10000` | A reasonable LoRA lands around 10k — that's where it *creatively underfits*: still varied on new prompts, not yet memorising. Past 10–20k it overfits. |
| **Batch size** | `1` on T4, up to `8` on H100 | Bigger = better gradients but more VRAM. |
| **Latent length** | model default, or shorter | You can train shorter than the model's max. Lower = faster, less memorization, useful for audio2audio. |
| **Learning rate** | leave default | Lower if you see grad-norm spikes. |
| **Demo every** | `500`–`1000` | Frequency of demo audio generation during training. |
| **Checkpoint every** | `500` | Frequency of `.safetensors` saves. Each one is restartable. |

**Tips:**

- **Latent length is the underrated knob.** Lowering it to ~47 s or ~12 s (with `random_crop` on) is often the cleanest way to learn a style *without* memorisation. The model only ever sees patterns at that timescale and never sees full songs, so it can't memorise structure. Listen to the demos at full length to hear how it extrapolates.
- **Batch size is a creative parameter.** Just because you have VRAM for 8 doesn't mean you want 8. `batch_size=1` learns something different (focuses on one song at a time, sharper imprint) from `batch_size=4` (averages gradients across songs, smoother fit). Experiment.

### 5. Pick a GPU

Each GPU card shows current VRAM use + an estimate of how much your run will need (based on model + rank + batch). If the estimate goes red, lower batch size or rank. The wizard pins the chosen GPU via `CUDA_VISIBLE_DEVICES` automatically.

### 6. Configure prompts

How prompts get built from your dataset each training step. Three sources with balance percentages that should sum to 100:

- **Tags** — the metadata fields from Step 2 (album / year / genre / bpm / …). Toggle which keys to include.
- **Paths** — directory + filename. Useful when your folder structure encodes meaning (e.g. `metal/death/archspire/`).
- **Fixed** — a literal string applied to every file. Use for single-style datasets where every example should share the same prompt.

Other knobs:
- **Trigger word** — optionally prepend a magic phrase to N% of prompts. The LoRA learns to associate the phrase with your style; at inference time, including the trigger turns the style on. A short unique token like `mfd3` + `trigger_pct=80` works well.
- **Shuffle** — randomize the order of comma-separated parts each step (keeps the model from memorising comma order).

**Simplest config for a small single-style dataset:** Fixed text = your trigger phrase, balance 100 % fixed.

**Good mix to try:** ~50/50 fixed + tags (or fixed + paths). Training sees both the shared style anchor *and* per-song detail (title, BPM, genre, etc.). At inference you can either use just the trigger (style only) or trigger + per-song details (style on a specific kind of song).

**You're blending two prompt vocabularies.** SA3's base model was trained on labelled key-value strings like `Genre: techno, BPM: 140, Mood: dark`. Your LoRA prompts get composed *on top* of that vocabulary at inference time. Re-using the base model's format seems to help the LoRA stack with what the base already knows, but you can also combine old and new (e.g. `/metal/song.mp3, BPM: 140`). Skim a few of your demo prompts to spot-check the format.

### 7. Set up demos

The MP3s + spectrograms generated for you to listen to during training. You can:
- Edit individual demos: prompt text, sampler (RF or ARC), CFG, step count, seed.
- Re-roll prompts with the per-demo `↻` or the big `↻` to re-sample from the dataset.
- Add / remove demos with `+` / `−`.

### 8. Launch and what to watch

Click **Launch**. Then:
1. **Run appears** in the run panel with status `loading…` (5–15 s while the base model loads into VRAM).
2. Status flips to `training`; **loss curve** starts plotting.
3. Every `demo_every` steps the run pauses ~30 s to generate **demo MP3s** with tinted-stereo spectrogram previews.
4. Every `checkpoint_every` steps a fresh `.safetensors` lands in the **Checkpoints** list.

### 9. How to know when to stop

**Loss curve.** Watch the loss panel. The *elbow* — where loss stops dropping steeply and starts to flatten — tends to be a good checkpoint to keep. Past the elbow you're refining, but also creeping toward memorisation.

**Your ears, on the demos.** Open a demo MP3 every couple thousand steps. What you're listening for:

- **Base RF demos (CFG≈7) light up first.** Around the time the run is "getting it," your CFG=7 demos suddenly sound right — clearly your style on a coherent prompt.
- **Then CFG=7 over-cooks, and CFG=1 takes over.** Past the elbow, CFG=7 starts sounding artifacted / over-saturated. The lower-CFG demos (CFG≈1) keep improving and end up sounding cleanest. If CFG=1 sounds good and CFG=7 doesn't, that's a sign the LoRA has internalised the style and no longer needs the prompt-classifier guidance.
- **Conditional → unconditional crossover.** Early on, only prompted demos sound like the training style. Later, even **empty-prompt** demos start sounding like the style — the model has *absorbed* the dataset.
- **ARC demos lag a bit but end up cleaner.** ARC-distilled demos take a few thousand more steps to catch up to base RF, but final quality is usually better.

**Don't fear a memorized checkpoint.** "Overfitting" is only a problem if you're chasing creative variation. A memorized checkpoint can be incredibly useful:
- Dial **LoRA strength** down at inference (~0.7) for clean "in the style of" generations.
- **audio2audio / style-transfer** effects often hit harder with a memorized model — the strong style signal pulls input audio into the training distribution more decisively.

Save checkpoints liberally. Different downstream uses want different points on the underfit ↔ memorise curve.

You can stop a run anytime — the last checkpoint is yours. You can also resume from any checkpoint to train further.

### 10. Get the `.safetensors` out

In the dashboard's checkpoints list, click the download (⬇) button — or grab it directly from disk at `state/runs/<run-id>/<step>.safetensors`. The file is the standard SA3 LoRA layout, ~20 MB for DoRA/LoRA at rank 16 (DoRA-XS is smaller). Drop it into any SA3 inference setup and the adapter layer-grafts onto the base model at runtime.

---

## Run inference with your LoRA

### From the dashboard

Click 🚀 on any checkpoint. Spins up a private Gradio at `http://localhost:7860`. Knobs:

- **LoRA strength <1.0** — blends LoRA with base model. A heavily-trained / "memorised" checkpoint often sounds best around 0.6–0.8 — gives you "in the style of" without straight regurgitation.
- **LoRA interval (skip first step)** — skipping the LoRA on the very first denoising step lets the base model establish song structure before your style takes over. Counterintuitively this often prevents the LoRA from feeling muddy or static.
- **audio2audio** — drop in a song; SA3 + your LoRA stylize it.
- **Inpainting** — paint over a region of an input and regenerate just that region in the LoRA's style.

### From the CLI

```bash
.venv/bin/python run_gradio.py \
    --model-config state/models/sa3-medium/base/model_config.json \
    --ckpt-path    state/models/sa3-medium/base/model.safetensors \
    --lora-ckpt-path my-lora.safetensors                 # one or more
```

Pass multiple `.safetensors` to blend LoRAs. Strengths interact non-linearly — worth experimenting.

### Beyond the dashboard

- **Continue training from an uploaded LoRA.** In *New Finetune*, the *Start from a previous LoRA* upload picks up your `.safetensors` and keeps going. There's an art to mixing datasets this way — e.g. train a music-style DoRA (Dadabots, baroquecore) for 10k steps at 47s, then continue on a production-quality DoRA (Encanti, bass music) at 12s for 100 steps. The result is one LoRA that fuses both styles in a way neither alone could have produced.
- **Any SA3 inference setup can load it.** Plugins, ComfyUI nodes, custom Python wrappers — the file is the standard SA3 LoRA layout.

---

## Running on Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dada-bots/underfit/blob/main/underfit-colab.ipynb)

Same underfit, same dashboard, just hosted on Colab's GPU instead of yours. The notebook walks through everything end-to-end.

### Why Colab

Zero install on your machine. Free GPU access (T4) for experimenting. Pro tier ($10/mo) gets you H100 at ~1.6 compute credits/hr — about $9.99 of credits per session. Drive integration means your runs persist across sessions.

### Important Colab-specific quirks

**Use ngrok in Step 4.** Colab's built-in port-proxy buffers HTTP responses aggressively, which makes the dashboard feel laggy and occasionally freeze. Audio playback is the worst offender — the proxy holds the entire audio file before forwarding to your browser instead of streaming it, blocking every other request until done. *Training itself is unaffected* — it runs as a detached subprocess on the GPU and survives dashboard freezes or closed tabs. Free signup at [ngrok.com](https://ngrok.com) and paste your auth token into the `NGROK_AUTHTOKEN` field in Step 4. If anything ever freezes, re-run Step 4 to restart the server.

**Storage is split between Drive (slow, persistent) and local SSD (fast, ephemeral).**

| Lives on | What |
|---|---|
| **Drive** (`UNDERFIT_STATE_DIR`) | runs.json, datasets, audio, gradio logs — small files, must persist across session resets |
| **Local SSD** (`UNDERFIT_MODELS_DIR=/content/models`) | Model checkpoints used by training. Reads at ~500 MB/s vs Drive's ~30 MB/s. Wiped on session reset, re-fetched from HF in 2–3 min. |
| **Local SSD** (`HF_HOME=/content/hf-cache`) | HuggingFace download cache. Stays local because re-downloading from HF (~150 s) is faster than copying from a Drive cache (~500 s). |

**Closing the tab is fine.** Training is a detached subprocess. The dashboard re-reads run state from Drive on next launch — your run was still going the whole time.

**Colab idle timeout.** The Step 5 monitor cell runs blocking with periodic output, which keeps the kernel comm channel busy and the cell visibly running — Colab's "active session" heuristics like both. Not a perfect keepalive (Colab still eventually disconnects if the browser tab is truly idle for hours), but it helps.

**Step 4 is the restart button.** Re-running it kills the old dashboard subprocess and starts a fresh one. Training runs are unaffected.

### Tier recommendations

| Tier | Notes |
|---|---|
| **H100** (Pro) | Best. Training is fast, demos finish in seconds, dashboard responsive. ~$9.99 of compute credits per session. |
| **L4 / A100 / G4** | Comfortable. |
| **T4** (Free) | Slow but works. Use **SA3-small** (medium can work but it's slow), **latent length ≤128**, **batch_size 1**, and the **Two-demo preset**. |

### Colab troubleshooting

- **"NO GPU DETECTED" in Step 1** → Runtime → Change runtime type → Hardware accelerator → pick a GPU → Save. Re-run Step 1.
- **`401 Unauthorized` / `GatedRepoError` in Step 3** → Accept the SA3 license at https://huggingface.co/stabilityai/stable-audio-3-medium. One click unlocks all three ARC repos. Re-run Step 3.
- **Dashboard frozen** → re-run Step 4. Training runs are unaffected.
- **`./install.sh` says "all packs staged" but ARC is missing** → Fixed in a recent commit. Pull latest: re-run the Step 8 "Pull latest underfit code" cell, then re-run install.

---

## Troubleshooting (general)

- **`GatedRepoError: 403 Client Error`** on ARC downloads → You haven't accepted the [SA3 license](https://huggingface.co/stabilityai/stable-audio-3-medium). One click, instant approval, all three ARC repos unlock.
- **`ModuleNotFoundError: No module named 'stable_audio_3'`** → The SA3 sibling clone got moved or never installed. Re-run `./install.sh` — it'll re-clone and `uv pip install -e` it.
- **Dashboard says port already in use** → another instance is running. Find with `ps aux | grep server.py`, kill it, then `./run.sh` again.
- **Demos sound identical to the input by 5k steps** → overfitting too fast. Lower the rank or stop earlier.
- **Demos sound nothing like the input even at 10k+** → dataset too varied, too small, or LR too low. Tighten the dataset or raise LR slightly.

---

## Project layout

```
underfit/             CLI + training loop + backend abstraction
dashboard/            webapp — server.py + index.html + per-model registries
dataset_processing/   pre-encode + autotagger
lora_train.py         trainer entrypoint
run_gradio.py         inference entrypoint (used by the 🚀 button)
install.sh, run.sh    install + launch helpers
underfit-colab.ipynb  the Colab tutorial — same dashboard, hosted on Colab
```

`stable-audio-3/` (the SA3 backend) gets cloned next to `underfit/` during install. Multiple `underfit/` clones share one SA3 clone and one HF cache.

---

## Contributing

Issues and PRs welcome at [github.com/dada-bots/underfit/issues](https://github.com/dada-bots/underfit/issues).

When reporting a bug, paste the output of:

```python
from underfit.monitor import debug_info; debug_info()
```

That gives the maintainer everything useful — git HEAD, torch + CUDA build, GPU compute caps, the latest run's log + every sidecar, disk usage, an SA3 import check.

---

## Credits & license

[MIT](./LICENSE). underfit is a thin layer over:

- **[Stable Audio 3](https://huggingface.co/stabilityai/stable-audio-3-medium)** by Stability AI — the diffusion model being finetuned
- **[Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools)** — the original training infrastructure that the `sat` backend mirrors
- LoRA family papers: [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al. 2021), [DoRA](https://arxiv.org/abs/2402.09353) (Liu et al. 2024)
- [`uv`](https://github.com/astral-sh/uv) for dependency management

Built by [@dadabots](https://dadabots.com).
