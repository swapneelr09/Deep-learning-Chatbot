from gtts import gTTS
import os



def say(text):
	tts = gTTS(text=text, lang='bn')
	tts.save("good.mp3")
	os.system("mpg321 good.mp3")