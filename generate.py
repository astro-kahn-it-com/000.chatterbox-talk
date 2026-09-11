import os
import sys
from datetime import datetime
import soundfile as sf
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
PROMPT_FILE = os.path.join(SCRIPT_DIR, "prompt.txt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["HF_HOME"] = MODELS_DIR
os.environ["TORCH_HOME"] = MODELS_DIR

from chatterbox.tts import ChatterboxTTS

def parse_prompt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing prompt file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    exaggeration = 0.5
    reference_audio = ""
    dialogue_lines = []
    current_section = None

    for line in content.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("[") and line_clean.endswith("]"):
            current_section = line_clean[1:-1].upper()
        elif line_clean:
            if current_section == "EXAGGERATION":
                try:
                    exaggeration = float(line_clean)
                except ValueError:
                    exaggeration = 0.5
            elif current_section == "REFERENCE_AUDIO":
                reference_audio = line_clean
            elif current_section == "DIALOGUE":
                dialogue_lines.append(line_clean)

    return exaggeration, reference_audio, " ".join(dialogue_lines)

def main():
    print("=" * 60)
    print(" Chatterbox TTS Standalone Engine")
    print("=" * 60)

    exaggeration, reference_audio, dialogue = parse_prompt(PROMPT_FILE)
    ref_audio_path = os.path.join(SCRIPT_DIR, reference_audio) if reference_audio else None

    print(f"[Directing] Exaggeration : {exaggeration}")
    print(f"[Reference] Voice Audio  : {ref_audio_path}")
    print(f"[Dialogue]  Synthesis    : {dialogue}")
    print("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Loader] Loading Chatterbox onto {device.upper()}...")
    model = ChatterboxTTS.from_pretrained(device=device)

    print("[Pipeline] Synthesizing speech...")
    has_ref = ref_audio_path and os.path.exists(ref_audio_path)

    wav_tensor = model.generate(
        dialogue,
        audio_prompt_path=ref_audio_path if has_ref else None,
        exaggeration=exaggeration
    )

    if isinstance(wav_tensor, torch.Tensor):
        audio_np = wav_tensor.squeeze().detach().cpu().numpy()
    else:
        audio_np = wav_tensor

    sr = getattr(model, "sr", 24000)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_file = os.path.join(OUTPUT_DIR, f"chatter_{timestamp}.wav")

    sf.write(out_file, audio_np, sr)

    print("-" * 60)
    print(f"[SUCCESS] Audio generated and saved to:")
    print(f"          -> {out_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()