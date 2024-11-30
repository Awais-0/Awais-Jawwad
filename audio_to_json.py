import whisper
from difflib import SequenceMatcher, ndiff
import json

def transcribe_audio(audio_file, model_name="base"):
    model = whisper.load_model(model_name)
    return model.transcribe(audio_file, word_timestamps=True)

def get_best_matching_speaker(segment_text, conversation, current_index):
    highest_similarity = 0.0
    best_match_index = -1

    for i in range(current_index, len(conversation)):
        similarity = SequenceMatcher(None, segment_text.lower(), conversation[i][1].lower()).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_index = i

    # If no significant match is found, default to the current index
    if highest_similarity < 0.5:  # Adjust threshold as needed
        best_match_index = current_index

    return best_match_index, conversation[best_match_index][0]

def align_words_with_provided_text(transcription_text, reference_text):
    aligned_words = []
    transcription_words = transcription_text.split()
    reference_words = reference_text.split()
    diff = list(ndiff(transcription_words, reference_words))

    for word in diff:
        if word.startswith(" "):  # Matched word
            aligned_words.append(word[2:])
        elif word.startswith("-"):  # Word in transcription not in reference
            continue
        elif word.startswith("+"):  # Word in reference not in transcription
            if word[2:] not in aligned_words:  # Avoid duplicates
                aligned_words.append(word[2:])
    return aligned_words

def transcription_to_json_with_speakers_and_alignment(result, conversation):
    json_data = {"segments": []}
    speaker_index = 0  # Start with the first speaker
    used_indices = set()  # Keep track of used conversation lines

    for segment in result['segments']:
        speaker_index, speaker = get_best_matching_speaker(segment['text'], conversation, speaker_index)

        # Skip if this conversation line was already used
        if speaker_index in used_indices:
            continue
        used_indices.add(speaker_index)

        aligned_words = align_words_with_provided_text(segment['text'], conversation[speaker_index][1])

        words = segment['words']
        aligned_word_data = []
        word_index = 0

        for word in aligned_words:
            if word_index < len(words) and word == words[word_index]['word']:
                aligned_word_data.append({
                    "word": word,
                    "start_time": words[word_index]['start'],
                    "end_time": words[word_index]['end'],
                    "duration": words[word_index]['end'] - words[word_index]['start']
                })
                word_index += 1
            else:
                if word_index > 0:
                    prev_word = words[word_index - 1]
                    start_time = prev_word['end']
                    end_time = start_time + 0.2  # Assume 0.2 seconds for missing word
                else:
                    start_time = segment['start']
                    end_time = start_time + 0.2

                aligned_word_data.append({
                    "word": word,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time
                })

        segment_data = {
            "speaker": speaker,
            "start_time": segment['start'],
            "end_time": segment['end'],
            "text": " ".join([word["word"] for word in aligned_word_data]),
            "words": aligned_word_data
        }
        json_data["segments"].append(segment_data)

    return json_data

def save_json(data, file_name="output_transcription_with_speakers.json"):
    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"JSON file saved as '{file_name}'")

def audio_to_json_with_speakers(audio_file, conversation):
    result = transcribe_audio(audio_file)
    json_data = transcription_to_json_with_speakers_and_alignment(result, conversation)
    save_json(json_data)
