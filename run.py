import speech_recognition as sr
from predict import response
from gtts import gTTS
import os

r = sr.Recognizer()

with sr.Microphone() as source:
	print("Speak anything")
	audio = r.listen(source)
	text = r.recognize_google(audio)
	print("You: "+str(text))
	text = response(text)
	print("Bot: " + str(text))
	tts = gTTS(text=str(text), lang='en')
	tts.save("good.mp3")
	os.system("mpg321 good.mp3")
