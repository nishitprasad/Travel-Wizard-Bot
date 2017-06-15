#Import necessary packages
import os, json, nltk, gensim, numpy as np, codecs, pandas as pd, theano, timeit
from gensim import corpora, models, similarities
from keras.models import Sequential
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.layers import Dense, Activation, TimeDistributed, Dropout
from sklearn.model_selection import train_test_split

#navigate to the respective project directory
os.chdir("/Users/nishit.prasad/Documents/NLP");

#Give specifications
outputDimension = 300   #Word to Vector Dimension
sentenceLength = 20     #Each conversation length


#Extract Conversation data from Frames dataset
with open('frames.json', 'r') as dataFile:
    data = json.loads(dataFile.read())

#Hold 2D conversation data list for each use case present in frames.json
conversationTextList = []
#1D list to hold all the conversation data present in frames.json
corpus = []

#Append necessary data in the respective lists
for i in range(len(data)):
    conversationListPerCase= []     #Hold conversation data per case in a list
    for j in range(len(data[i]['turns'])):
        sentence = data[i]['turns'][j]['text']
        conversationListPerCase.append(sentence)
        corpus.append(sentence)
    conversationTextList.append(conversationListPerCase)

#Necessary Word Tokenization
corpusTokens = [nltk.word_tokenize(sentence) for sentence in corpus]

#Converting respective words to vectors using gensim.model.Word2Vec (library)
wordToVector = gensim.models.Word2Vec(corpusTokens, min_count=1, size = 300) 

x = []  # conversation set except last line
y = []  # conversation set except first line

#Necessary implementation to fill x and y lists
for i in range(len(conversationTextList)):
    for j in range(len(conversationTextList[i])):
        if j<len(conversationTextList[i])-1:
            x.append(conversationTextList[i][j]);   # does not take last line
            y.append(conversationTextList[i][j+1]); # does not take first line

#To hold tokenized words present in the list x and y
tokensX=[]
tokensY=[]

#Word tokenization
for i in range(len(x)):
    tokensX.append(nltk.word_tokenize(x[i].lower()))
for i in range(len(y)):
    tokensY.append(nltk.word_tokenize(y[i].lower()))

#Fillers with random word vectors
sentend=np.ones((outputDimension,),dtype=np.float32)

#Converting respective words to vectors
vectorX=[]
for sent in tokensX:
    sentVector = [wordToVector[w] for w in sent if w in wordToVector.wv]
    vectorX.append(sentVector)
    
vectorY=[]
for sent in tokensY:
    sentVector = [wordToVector[w] for w in sent if w in wordToVector.wv]
    vectorY.append(sentVector)    

#Append necessary fillers   
for tokensSentence in vectorX:
    tokensSentence[sentenceLength-1:]=[]
    tokensSentence.append(sentend)
    
for tokensSentence in vectorX:
    if len(tokensSentence)<sentenceLength:
        for i in range(sentenceLength-len(tokensSentence)):
            tokensSentence.append(sentend)    
            
for tokensSentence in vectorY:
    tokensSentence[sentenceLength-1:]=[]
    tokensSentence.append(sentend)
    

for tokensSentence in vectorY:
    if len(tokensSentence)<sentenceLength:
        for i in range(sentenceLength-len(tokensSentence)):
            tokensSentence.append(sentend)             

##################### TRAIN THE MODEL ####################

#Convert the following lists to numpy lists
vectorX=np.array(vectorX)
vectorY=np.array(vectorY)     

#Get necessary training (80%) and testing data(20%)
xTrain, xTest, yTrain,yTest = train_test_split(vectorX, vectorY, test_size=0.2, random_state=1)

#Reshaping the training and testing data to fit the model
xTrain = xTrain.reshape(xTrain.shape[0], sentenceLength, outputDimension)
yTrain = yTrain.reshape(yTrain.shape[0], sentenceLength, outputDimension)
xTest = xTest.reshape(xTest.shape[0], sentenceLength, outputDimension)
yTest = yTest.reshape(yTest.shape[0], sentenceLength, outputDimension)

#Sequential Model considered for training, testing and prediction.
model=Sequential()

#Add necessary layers
model.add(LSTM(outputDimension,input_shape=xTrain.shape[1:],return_sequences=True, activation='sigmoid'))
model.add(LSTM(outputDimension,input_shape=xTrain.shape[1:],return_sequences=True, activation='sigmoid'))
model.add(LSTM(outputDimension,input_shape=xTrain.shape[1:],return_sequences=True, activation='sigmoid'))
model.add(LSTM(outputDimension,input_shape=xTrain.shape[1:],return_sequences=True, activation='sigmoid'))
model.add(Dense(outputDimension))
#Compile the model
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(xTrain, yTrain, epochs=20,batch_size=1, verbose=2,validation_data=(xTest, yTest))

#Evaluate the model
score = model.evaluate(xTest, yTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##################### TEST THE MODEL ####################

#While the conversation lasts (as per the user)
while(True):
    #Enter user's input
    x=input("Your message:")
    
    #Set up fillers
    sentend=np.ones((outputDimension,),dtype=np.float32) 
    
    #Tokenize the input sentence.
    sent = nltk.word_tokenize(x.lower())
    
    #Convert the words to vectors with reference to the wordToVector
    sentvec = [wordToVector[w] for w in sent if w in wordToVector.wv]

    #Append the fillers
    sentvec[sentenceLength-1:]=[]
    sentvec.append(sentend)
    if len(sentvec)< sentenceLength:
        for i in range(sentenceLength-len(sentvec)):
            sentvec.append(sentend) 
    
    #Convert list to numpy list
    sentvec=np.array([sentvec])
    
    #Predict the output
    predictions = model.predict(sentvec)
    outputlist=[wordToVector.most_similar([predictions[0][i]])[0][0] for i in range(sentenceLength)]
    output=' '.join(outputlist)
    print(output)