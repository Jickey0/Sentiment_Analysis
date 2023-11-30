Project Title: Live Mood Based Sentiment Analysis with Machine Learning


Author: Jack Hickey
Date: November 28, 2023


Description:
    The program records live audio and outputs an analysis of the speaker's 
mood via a machine learning model. Upon running the main file, a GUI is created, 
using PySimpleGUI, which asks them if they would like to record. Upon clicking 
the button the recording begins. First, we initialize the microphone using the 
imported speech_recognition tool. Then, we use Google to recognize audio and 
then convert the speech to text using gTTS. That speech is passed into our 
neural network, trained on IMDB’s movie review, to predict the mood of our 
speaker. When the key word “stop recording” is spoken, the recording ends. The 
data recording is then converted into a happiness over time graph with matplotlib 
and displayed to the user. 
    This project serves as both a proof of concept and a tool for someone who wants 
to be more self conscious about the things they are saying online. By using this 
project we encourage users to reflect on the impact of their words. Ultimately, 
the goal is to increase a more positive environment online through higher self awareness.


Dependencies:
External libraries or modules include:
scikit-learn
imbalanced-learn
SpeechRecognition
gTTS
numpy
pandas
matplotlib
PySimpleGUI
pyaudio
joblib


Installation:
    Please run “pip install -r requirements.txt” to ensure you have all the necessary 
dependencies to run the program. 


Usage:
    It is recommended that you run the programs ‘IMDB_Model.py' and ‘main.py’ within the terminal 
window! Run “python main.py” to execute the program, however if you're experiencing errors with 
microphone permissions try “sudo python main.py”. 



Example:
    Here is some example speech inputted into the program! 

    "Let me tell you a story… So I got up this morning and there was some fantastic weather and 
honestly I thought today was going to be great. However, then my day took a turn for the worse, 
I found out that squid coin was actually a scam and I had lost all my money because chat gpt 
told me to invest. But then I bought a lottery ticket with the last bit of money I had and won! 
It was fantastic and I couldn’t have been happier. Maybe the real squid coin was the friends we 
made along the way…"

    That was a fantastic story, but here is how the program used that story. 
First, after we pressed the record button, the program converts our speech into text live. After 
the key words “stop recording” are said, a graph representing the probability over time is generated. 
Additionally, our entire transcript is used to create an overall metric for happiness which is 
printed to the screen. To see these examples please look at the MyExampleGraph.png and MyExampleGUI.png 
files located inside of the images folder. 

Project Structure:
    The project is composed of two major components. IMDB_Model.py which creates our neural network and 
main.py which handles the GUI, transcription, and prediction of the users speech. Both of these files 
are kept in the main directory. 

    The IMDB_Model.py creates both our neural network, IMDB_log_reg.joblib, and our TF-IDF based 
vectorizing function, TFIDF_Vectorize.npy. The data used to train our model is found in 
data_sets/IMDB Dataset.csv. This is a free to use, pre labeled, dataset containing 50k movie 
reviews and their corresponding evaluations of that movie. The dataset is kept in the folder 
“data_sets” and the models are saved into the folder “models”.

    Additional files such as README.txt and requirements.txt are kept in the main directory as well. 
However, images such as tomato.png and our two example images, MyExampleGraph.png and 
MyExampleGUI.png, are kept in the images folder. 


Files:
IMDB_Model.py: 
    For this project we used natural language processing techniques to create a neural network 
trained with scikit-learn’s logistic regression model. The IMDB movie review database from kaggle 
was chosen for the training process. The idea was that the language associated with positive 
reviews would correlate to positive emotions in a speaker. So, by creating a model that can 
guess if a movie review was “positive” or “negative” we essentially create a universal semantic 
analysis tool. While, often there is some context necessary to understand and evaluate speech, 
this should work well enough. 
	The primary tools leveraged include pandas, scikit-learn, and imbalanced-learn. The code 
begins by importing the necessary libraries and reading an IMDb dataset from a CSV file using 
pandas. Then, it separates the reviews into positive and negative sentiments, concatenates them, 
and applies Random Under-Sampling to address potential overfitting. We then resample the dataset 
and separate once again. For more information of avoiding overfitting please see: 
https://statisticsbyjim.com/regression/overfitting-regression-models/

    Next, the code employs the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization 
technique to convert the text data into numerical features. Using a linear regression model 
our model is trained using the training dataset. Lastly, the TF-IDF vectorize function and 
the trained logistic regression model are saved to be used later in main.py.
For more information on TF-IDF please see: https://monkeylearn.com/blog/what-is-tf-idf/ 


Main.py:
    First we load imports in our dependencies we will be using for the project. Important tools 
such as speech_recognition and gtts are used for speech to text, PySimpleGUI for the user 
interface, and matplotlib for the graphing. 

    Then we load in our pre-trained models TFIDF_Vectorize.npy and IMDB_log_reg. Next we define our 
global variables such as our time, the microphone recognizer, our total transcribed message, and 
our final happiness score. 

    PySimpleGUI is used to create a simple GUI for the user to display a canvas with some descriptive 
text, a record button, an exit button, and a fun movie review inspired tomato image. This layout 
is saved into a variable called “window”. 

    The file main function performs the event based logic for the values within our window. After 
rendering the values of the window to the screen, we define an event, a button click, for 
recording. When called we use the function speechToMood to start the sentiment analysis process. 

    The speechToMood function first defines our x and y values which will represent our happiness 
and the time elapsed. Then we define a string which will contain all of the words used in the 
recording. Then we will infinitely loop through a recording trying to record using the microphone 
and catching exceptions if no words are said. During each loop we pause for 0.15 seconds to adjust 
for ambient sound, thus improving the audio quality with a small chance of missing words or 
speech during the adjustment time. We use speech recording’s adjust_for_ambient_noise function for 
this and the listen function to actually activate the microphone. Next we use google’s speech to 
text API to convert our recording into text. Additionally, we make sure to put all words into 
lowercase for the models. 

    If the key words “stop recording” are said, we end the session and print "session stopped" in 
the terminal. However, if the key words are not said, then print the user's text in the terminal. 
Then the text is sent to the TFIDF_Vectorize.npy, to convert the words into vectors to represent 
our text in numerical format. The program then returns the matrix composed of our word vectors, 
which we feed into the neural network. IMDB_log_reg.joblib performs the calculation and then 
sends back a number between 1 and 0, representing the probability that a message is either 
positive or negative. 

    The timestamps are saved into mood_data_x by taking the difference between now and the start 
of the program. On the other hand, the probability for a happy review is saved into the 
mood_data_y. The function makes one final prediction using all of the text transcribed in the 
text and saves it as final_total_prediction. Then speechToMood returns mood_data_x, 
mood_data_y, and final_total_prediction. 

    The main function now can print out our final happiness percent by updating the window. 
Additionally, we create a plot using the function create_plot. The function takes in an 
X and Y list as parameters and generates a plot using matplotlib. Then, our desired location 
for the data to be stored in the canvas and our new plot are sent into the show_figure 
function. This adds our figure to our canvas using the FigureCanvasTkAgg() function, thus 
displaying it to the user. 

Then the sentiment analysis process is complete! 


Future Improvements:
    There are several changes that could be added to increase performance, user experience, 
and overall functioning of the code. First I would like to add a more specific mood 
evaluation. For example, not just positive or negative, but maybe additional emotions like 
happy, tired, sad, etc. Next, I would like improving the model by maybe more layers or 
combining different databases. The database is trained only in the context of movie review 
which can give us some inaccurate results depending on the context of the text. For example, 
we would associate a movie being “boring and predictable” as a negative, however the stock 
market being “predictable” would likely be positive. Lastly, I would like to improve the 
speed of the program as it can occasionally run fairly slow depending on the device used.

    Additionally, my computer's microphone is not that great, so getting a better one would most 
likely improve the accuracy of the speech to text API.


Sources:
Pysimplegui:
1. https://www.tutorialspoint.com/pysimplegui/pysimplegui_matplotlib_integration.htm
2. https://www.tutorialspoint.com/pysimplegui/pysimplegui_events.htm

model:
3. https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
4. https://huggingface.co/blog/sentiment-analysis-python
5. https://www.analyticsvidhya.com/blog/2022/07/sentiment-analysis-using-python/
6. https://www.youtube.com/watch?v=QAZc9xsQNjQ&t=1s&ab_channel=CS50
7. https://imbalanced-learn.org/stable/over_sampling.html 

Speech to Text:
8. https://www.geeksforgeeks.org/convert-text-speech-python/
9. https://www.geeksforgeeks.org/python-convert-speech-to-text-and-text-to-speech/#
10. https://pypi.org/project/gTTS/

Images:
11. https://www.clipartmax.com/middle/m2i8H7G6G6m2G6b1_rotten-tomatoes-fresh-logo/ 

More Info:
12. https://statisticsbyjim.com/regression/overfitting-regression-models/
13. https://monkeylearn.com/blog/what-is-tf-idf/ 