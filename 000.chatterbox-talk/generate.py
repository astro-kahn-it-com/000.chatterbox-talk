import os
import time
import soundfile as sf
from chatterbox.tts import ChatterboxTTS

def parse_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    exaggeration = 0.0
    reference_audio = ""
    dialogue = ""

    current_section = None
    for line in content.splitlines():
        line = line.strip()
        if line == "[EXAGGERATION]":
            current_section = "EXAGGERATION"
        elif line == "[REFERENCE_AUDIO]":
            current_section = "REFERENCE_AUDIO"
        elif line == "[DIALOGUE]":
            current_section = "DIALOGUE"
        elif line:
            if current_section == "EXAGGERATION":
                exaggeration = float(line)
            elif current_section == "REFERENCE_AUDIO":
                reference_audio = line
            elif current_section == "DIALOGUE":
                dialogue = line

    return exaggeration, reference_audio, dialogue

def main():
    # Force cache directories to point to local models/ directory
    os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")
    os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), "models")

    # Parse prompt.txt
    exaggeration, reference_audio, dialogue = parse_prompt("prompt.txt")

    # Initialize model
    model = ChatterboxTTS.from_pretrained(device="cuda")

    # Generate audio
    result = model.generate(dialogue, reference_audio, exaggeration)

    # Handle possible return types from model.generate()
    if isinstance(result, tuple):
        audio_data, sample_rate = result
    else:
        audio_data = result
        sample_rate = 24000  # Default fallback sample rate

    # Save resulting .wav file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("output", f"output_{timestamp}.wav")
    sf.write(output_path, audio_data, sample_rate)

if __name__ == "__main__":
    main()
