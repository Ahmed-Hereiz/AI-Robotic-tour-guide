from gtts import gTTS
import pyaudio
import os
from customAgents.agent_llm import SimpleStreamLLM
import json

with open("config/llm.json", "r") as f:
    llm_config = json.load(f)

llm = SimpleStreamLLM(api_key=llm_config['api_key'], model=llm_config['model'], temperature=0.3)
text = llm.generate_response("hello")
tts = gTTS(text, lang='en')

# Save the audio to a temporary file
temp_audio_file = "temp_audio.mp3"
tts.save(temp_audio_file)

# Play the audio file
os.system(f"mpg321 {temp_audio_file}")  # Install mpg321 using: sudo apt-get install mpg321

# Clean up the temporary file
os.remove(temp_audio_file)
# Save the output as a permanent MP3 file
output_filename = "output_audio.mp3"
tts.save(output_filename)
print(f"Audio saved as {output_filename}")
