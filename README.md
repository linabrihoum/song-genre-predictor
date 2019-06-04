# Decibels
## Problem
A problem in the world is not knowing information about a song, so being able to predict that information is helpful for recommending it to people who like songs that are in the same genre. The constant advancement of technology regarding music entertainment brings up these demands. With our program we solved both problems by determining the genre of a song through machine learning algorithms and analysis of sound waves. This might be used to enable more accurate predictability and recommendation to be used by consumers and producers in the future.
## Solution
Our solution to the problem mentioned above was solved using machine learning and a lot of publicly available music from our own playlists and from other datasets online. We trained a model to let the user know the genre and in the future this could be extended to include the artist, country of origin, or even the particular song if we apply this to database searching. In this project we used various audio analysis techniques to get useful inputs for our model. These are more helpful than the raw audio input would otherwise be since they are designed to show particular information about a song. Our model can provides useful information to potential consumers and allows for more accurate music plays for mood fitting in various situations. 
## Data & Analysis
There are many ways to visualize an audio waveform in this project through the librosa library, we visualize the spectrogram, frequency axis, zero crossing rate, spectral centroid, time variable, spectral rolloff, mel-frequency cepstral coefficient, and chroma feature for every song. 

To begin, we analyzed a single audio file to create a basis for the rest of our audio files. Sound is represented in the form of sound waves and signals, hence why we are using .wav files to analyze the audio file, however, our program can analyze any audio file. All of the work we preformed is using Jupyter Notebook and Python as well as the main library import called librosa which is able to analyze audio waves using different features and graphs.

The first analysis performed was on a audio file to visualize the sound wave. It is represented below. When our program classifies a song, it will cut the audio into 30 seconds, however, to visualize the graphs we analyzed the entirety of its song. 

After visualizing the initial amplitude of the waveform, we started to really analyze the waveforms through various graphs. The first graph we used is a spectogram. A spectrogram is a visual representation of the spectrum of frequencies of sound as they vary with time. The vertical axis shows frequencies, and the horizontal axis shows the time of the clip.

There are a number of data points used as features for the model. One of these is the zero crossing rate, which is the number of times an audio signal crosses the zero point. This can be seen by zooming into an audio waveform. This parameter is useful for genre detection since different musical genres will have different zero crossing rates. Knowing the zero crossing rate, and how that changes throughout an audio track is a good way of helping to nail down its genre.

The spectral centroid is another data point used for our model. The spectral centroid is particularly useful at helping to determine the timbre of an audio track. For music the timbre, broadly speaking, is the sound of a song. It is what helps to separate the sounds of various instruments in the same category. This is extremely helpful for our application since musical genres are often defined around the kinds of instruments that can be heard in the songs.

Mel-Frequency & Frames are another kind of algebraic process done on the audio signal to give a useful parameter for genre classification. It helps to show how many frames there are. The example below shows all of the features for a song around 
6 minutes in duration. The length of this clip is only for demonstration purposes, all of the audio samples that we used for training and testing were only 30 seconds long. This is also the last graph we visualize before performing the same process to the entire dataset of songs.

After all the graphs are constructed for a single clip, we then graphed all the audio files in our dataset. Using the GTZAN dataset, our program visualized every audio file and made a new folder with all the graphs for each song and genre.

Once all the graphs are processed, we then converted the graphs to a numerical representation and put all the data in a CSV. Once all the data is in a CSV file, it is easier to train and test.

After our CSV file is processed with all the numeral data, we encoded the labels and scaled the features. After all this is done, the training and testing took place. We then fit the model using our training data and checked the accuracy of our model. Our training model accuracy came out to be around 80-90% while our testing accuracy was around 70%. 

## Accuracy
Our accuracy of the model is at 94% and our testing accuracy is 70%. The accuracy fluctuates depending on the hyperparameters. Originally, the epochs (iterations) was only 100 steps which resulted in a low accuracy of 60%, however, by increasing the iterations to around 10,000 or 100,000 drastically increased the accuracy. Also, in increasing the iterations our program was able to cross validate itself to further improve the accuracy.

## Algorithms
The two main algorithms we used are Random Forest and Neural Networks. 

Random Forest is a highly accurate classifier ensemble algorithm we tested that is specifically for classification and regressions which worked fairly well for our program. Random Forest improves the predictive accuracy and controls over-fitting. When we implemented this algorithm our main goal was to make the program more accurate than the previous tests. Using the bagging method, the combination of models increased the result accuracy overall. The use of Random Forest helped predict the genre by implementing various features and their importance through random decision trees to further increase the accuracy of our prediction model.

Neural Networks are a set of algorithms, modeled loosely to recognize patterns. By using neural networks, it has helped our program recognize the patterns of the different audio features and graphs and in thus, predicting the genre with a very high accuracy score. Our program uses 3 hidden layers and a total of 800 neurons. Our team had to brute force to find the best way to split up the neurons to determine the best accuracy. By splitting the neurons up by 500, 200, and 100 for their respective layer, it resulted in an accuracy of 94%. We also tested our program using 1000, 10,000, and 100,000 iterations.

## Results & Predictions
Currently we are able to get a song genre prediction with around an accuracy of 94%. Improving this accuracy would be nice but this is a tricky task. The many different variables and factors that make songs unique is also making it difficult to accurately predict a songs genre. One way this could be achieved is through the introduction of metadata. Since most practical applications of what we’re doing would have access to valuable information outside of what we could get by just analysing the audio we considered this as a way of improving accuracy. In the end we decided against this since we felt we were able to get our accuracy up high enough without it, and being able to analyze a song just based on an audio signal is valuable in its own right.

The way our program is able to predict the songs genre is by splitting it up in 30 second intervals and predicting each 30 second intervals genre and then come to the conclusion at the end by the most popular predicted genre. It determines the genre by the features however, we rated each feature by importance and how important each feature should play a role in determining the genre.

The program will read the audio file and apply every feature and audio analysis to the single song file and print out the numbers associated with each feature.

Once the features are extracted, it will predict every 30 seconds and determine the genre at the end depending on the most popular genre. The way it is able to predict the genre is by comparing the features of the song it’s trying to predict to the dataset. 

## Conclusion

In conclusion, we were able to successfully predict the genre of a song with good accuracy using just analysis of the waveform. We can of course improve this accuracy even more by adding meta data analysis, which would also allow us to extend what this program can predict.
