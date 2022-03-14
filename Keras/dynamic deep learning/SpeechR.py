import speech_recognition as sr
print(sr.__version__)
print("Now listening...")

r = sr.Recognizer()
mic = sr.Microphone(device_index=0)

with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

r.recognize_google(audio)
print(r.recognize_google(audio))
#sr.Microphone.list_microphone_names()
