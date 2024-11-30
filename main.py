from gradio_client import Client
from text_to_audio import generate_audio_from_text
from audio_to_json import audio_to_json_with_speakers

def generate_dialogue():
    client = Client("jawwad1234/convo")
    result = client.predict(
        message="""create a very short hypothetical dialogue between student and teacher about astronomy """,
        system_message="""You are a chatbot which only replies shortly and in the form of a dialogue 
        between two characters talking about given topic in just 10 lines, their names are john and smith, when starting
        each person's dialouge, only start it with: '[john]:' or '[smith]:' if you have to mention the names again, don't use brackets, 
        brackets only come when starting the sentence
        and no need to mention names or any kind of alias for them. """,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        api_name="/chat"
    )
    return result

def extract_conversation_from_text(text):
    """
    Extracts conversation into a structured list format.
    Example:
    Input: "[Person 1] Hello.\n[Person 2] Hi there."
    Output: [("Person 1", "Hello."), ("Person 2", "Hi there.")]
    """
    lines = text.split("\n")
    conversation = []
    for line in lines:
        if line.strip():
            parts = line.split("]", 1)  # Split on the first occurrence of `]`
            if len(parts) == 2:
                speaker = parts[0].strip("[ ").strip()
                sentence = parts[1].strip()
                conversation.append((speaker, sentence))
    return conversation

def main():
    print("Generating dialogue...")
    dialogue = generate_dialogue()
    print("Generated dialogue:")
    print(dialogue)

    # Extract structured conversation from the generated dialogue
    conversation = extract_conversation_from_text(dialogue)

    # Generate audio from text
    final_audio_path = generate_audio_from_text(dialogue)

    # Create JSON from audio using the extracted conversation
    audio_to_json_with_speakers(final_audio_path, conversation)

if __name__ == "__main__":
    main()
