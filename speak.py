''' Script to test speak python 
    Author: Jedid Santos
    E-mail: jedid.santos@gmail.com
    Date: March 03th, 2021
    Masters Engineering University of Algarve
'''

import pyttsx3
engine = pyttsx3.init()
engine.say("Jedid")
engine.runAndWait()