from gtts import gTTS 
import numpy as np
import scipy
import soundfile as sf
from scipy.io.wavfile import write
#from scipy.io.wavfile import read
import simpleaudio as sa

#text = input("Enter text: ")
text="Hi there, this is GladBaby! How are you?"

# Generate speech 
tts = gTTS(text=text, lang='en')
tts.save("speech.wav")

# Load audio data 
data, rate = sf.read("speech.wav")

# Apply whisper filter
whisper_data = np.interp(np.arange(0,len(data)), np.arange(0,len(data)), data).astype(data.dtype)
scaled = whisper_data / np.max(np.abs(whisper_data)) * 0.5 

# Save modified speech
sf.write("processed.wav", scaled.astype(np.int16), rate)
# Write back to file
#sf.write('processed.wav', data, rate)
# Play whisper speech
#wave_obj = sa.WaveObject(scaled, 1, 2, rate)
#play_obj = wave_obj.play()
#play_obj.wait_done()

