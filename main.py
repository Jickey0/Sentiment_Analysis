import speech_recognition as sr
from gtts import gTTS
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg

# load in our models 
tfidf = np.load('TF-IDF.npy', allow_pickle=True).item()
log_reg = np.load( 'IMDB_log_reg.npy', allow_pickle=True).item()

# keeps track of our starting time
start_time = time.time()

# Initialize the recognizer
r = sr.Recognizer()

# define our GUI
layout = [
    [sg.Canvas(size=[400,400], key='-CANVAS-')],
    [sg.Text('Analyze your speech! Press record and say the key word "stop recording" to end the session')],
    [sg.Button("Record", key='-RECORD-'), sg.Exit()],
]
# create our window
window = sg.Window('Sentiment Analysis', layout)

def main():
    # display window and preform event logic
    while True:
        event, values = window.read()
        print(event, values)

        if event == "-RECORD-":
            x_and_y_data = speechToMood()
            
            # define our x's and y's
            mood_data_x = x_and_y_data[0]
            mood_data_y = x_and_y_data[1]

            # create figure / plot
            my_figure = show_figure(window['-CANVAS-'].TKCanvas, create_plot(mood_data_y, mood_data_x))

        if event == "-SUB-":
            result = int(values['-FIRST-']) - int(values['-SECOND-'])

        # TODO: add live text to speech when talking
        # window['-OUT-'].update(result)

        if event == psg.WIN_CLOSED or event == 'Exit':
            break
    window.close()

# --- main abstracted recording function -- #

def runProgram():
    x_and_y_data = speechToMood()
    mood_data_x = x_and_y_data[0]
    mood_data_y = x_and_y_data[1]

    my_figure = show_figure(window['-CANVAS-'].TKCanvas, create_plot(mood_data_y, mood_data_x))
    return

# --- helper functions -- #

# Function to convert text to speech using gTTS
def SpeakText(command):
    tts = gTTS(text=command, lang='en')
    tts.save("output.mp3")
    os.system("afplay output.mp3")  # For macOS, plays the saved audio file

def speechToMood():
    # define our lists to contain happiness and time
    mood_data_x = []
    mood_data_y = []
    
    # define string to contain all words said
    all_words = ""
    
    # Loop infinitely for the user to speak
    while True:
    # Exception handling to handle exceptions at runtime
        try:
            # Use the microphone as the source for input.
            with sr.Microphone() as source:
                # Wait for a second to let the recognizer adjust the energy threshold based on the surrounding noise level
                r.adjust_for_ambient_noise(source, duration=0.2)
                
                # Listen for the user's input
                audio = r.listen(source)
                
                # Using Google to recognize audio
                MyText = r.recognize_google(audio)
                MyText = MyText.lower()
                
                if "stop recording" in MyText:
                    print("session stopped")
                    break
                
                print("hellow?")
                print("Did you say: " + MyText)
                
                input_vector = tfidf.transform([MyText])
                
                # Get predicted probabilities for each class
                predicted_probabilities = log_reg.predict_proba(input_vector)
                
                # Make final predictions
                final_predictions = log_reg.predict(input_vector)
                print("Final Predictions:", final_predictions)
                
                # find the diffence between now and our start time to add to mood_data_x
                end_time = time.time()
                mood_data_x.append(end_time - start_time)
                
                mood_data_y.append(predicted_probabilities[0][1])
                all_words += MyText + " "
                
        except sr.RequestError as e:
            print("Could not request results:", e)
            
        except sr.UnknownValueError:
            print("say something")
            continue
    
    return (mood_data_x, mood_data_y)

def create_plot(happiness, time):
    # plot probability data from recording
    plt.plot(time, happiness)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Happiness (kilo-smiles)")
    plt.show()
    return plt.gcf() # returns our graph as a figure

def show_figure(fig, canvas):
    fc_agg = FigureCanvasTkAgg(fig, canvas)
    fc_agg.draw()
    fc_agg.get_tk_widget().pack(side='top', fill='both', expand=1)



#plt.show()
#final_predictions = log_reg.predict(input_vector)
#print("Overall your story was... ", final_predictions)

if __name__ == "__main__":
    main()