import numpy as np
import onnxruntime
import soundfile as sf
import yaml
import os
import re
from ttstokenizer import TTSTokenizer
from pydub import AudioSegment

# Load configuration
with open("vctk-vits-onnx/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Create model
model = onnxruntime.InferenceSession(
    "vctk-vits-onnx/model.onnx",
    providers=["CPUExecutionProvider"]
)

# Create tokenizer
tokenizer = TTSTokenizer(config["token"]["list"])

# Define speaker IDs for Person 1 and Person 2
person_1_sid = 17  # Speaker ID for Person 1
person_2_sid = 14  # Speaker ID for Person 2

def text_to_conversation_with_tags(text):
    pattern = r"\[([^\]]+)\]\s*(.*)"
    matches = re.findall(pattern, text)
    return [(speaker.strip(), sentence.strip()) for speaker, sentence in matches]

def generate_audio_from_text(text, output_path="final_conversation.wav"):
    conversation = text_to_conversation_with_tags(text)
    output_dir = "conversation_audio"
    os.makedirs(output_dir, exist_ok=True)

    # Generate speech for each line
    for i, (speaker, line_text) in enumerate(conversation):
        sid = person_1_sid if speaker == "John" else person_2_sid
        inputs = tokenizer(line_text)
        outputs = model.run(None, {"text": inputs, "sids": np.array([sid])})
        output_file = os.path.join(output_dir, f"line_{i}.wav")
        sf.write(output_file, outputs[0], 22050)
        print(f"Generated: {output_file}")

    # Concatenate the audio files into a full conversation
    conversation_audio = AudioSegment.empty()
    audio_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".wav")])

    for file in audio_files:
        conversation_audio += AudioSegment.from_file(file)

    conversation_audio.export(output_path, format="wav")
    print(f"Final conversation audio saved as: {output_path}")

    # Cleanup
    if os.path.exists(output_path):
        for file in audio_files:
            os.remove(file)
        os.rmdir(output_dir)
        print(f"Deleted directory: {output_dir}")

    return output_path
