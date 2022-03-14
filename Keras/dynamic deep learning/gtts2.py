
from gtts import gTTS
import os

mytext = 'what is this object?'

language = 'en-ng'
myobj = gTTS(text=mytext, lang=language, slow=False)

"""# Chinese
'zh-cn': 'Chinese (Mandarin/China)',
'zh-tw': 'Chinese (Mandarin/Taiwan)',
# English
'en-us': 'English (US)',
'en-ca': 'English (Canada)',
'en-uk': 'English (UK)',
'en-gb': 'English (UK)',
'en-au': 'English (Australia)',
'en-gh': 'English (Ghana)',
'en-in': 'English (India)',
'en-ie': 'English (Ireland)',
'en-nz': 'English (New Zealand)',
'en-ng': 'English (Nigeria)',
'en-ph': 'English (Philippines)',
'en-za': 'English (South Africa)',
'en-tz': 'English (Tanzania)',
# French
'fr-ca': 'French (Canada)',
'fr-fr': 'French (France)',
# Portuguese
'pt-br': 'Portuguese (Brazil)',
'pt-pt': 'Portuguese (Portugal)',
# Spanish
'es-es': 'Spanish (Spain)',
'es-us': 'Spanish (United States)' """

myobj.save("welcome.mp3")
os.system("mpg321 welcome.mp3")
os.remove("welcome.mp3")
