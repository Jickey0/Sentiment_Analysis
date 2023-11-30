Live Mood Based Sentiment Analysis with Machine Learning by Jack Hickey

Preface: 
Please run “pip install -r requirements.txt” to ensure you have all the 
necessary dependencies to run the program. 

'TF-IDF.npy' and 'IMDB_log_reg.npy' have been provided pre-trained for you, 
however feel free to train your own model using the file ‘IMDB_Model.ipynb.

It is recommended that you run the program with either VS-Code and/or in the 
terminal window! Run “python main.py” to execute the program, however if you're 
experiencing errors with microphone permissions run “sudo python main.py”. 


Introduction:
    The program records live audio and outputs an analysis of the speaker's mood 
via a machine learning model. Upon running the main file, a GUI is created, 
using PySimpleGUI, which asks them if they would like to record. Upon clicking 
the button the recording begins. First, we initialize the microphone using the 
imported speech_recognition tool. Then, we use Google to recognize audio and then 
convert the speech to text using gTTS. That speech is passed into our neural network, 
trained on IMDB’s movie review, to predict the mood of our speaker. When the key 
word “stop recording” is spoken, the recording ends. The data recording is then 
converted into a happiness over time graph with matplotlib and displayed to the user. 
    This project serves as both a proof of concept and a tool for someone who wants
to be more self conscious about the things they are saying online. By using this 
project we encourage users to reflect on the impact of their words. Ultimately, 
the goal is to increase a more positive environment online through higher self awareness.


IMDB_Model: 
For this project we used natural language processing techniques to create a neural network trained with scikit-learn’s logistic regression model. 


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
8. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sparse.from_spmatrix.html

Speech to Text:
9. https://www.geeksforgeeks.org/convert-text-speech-python/
10. https://www.geeksforgeeks.org/python-convert-speech-to-text-and-text-to-speech/#
