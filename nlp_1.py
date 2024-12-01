import pandas as pd
import numpy as np
import re
import transformer 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

train_data = pd.read_excel("Datasets/train.xlsx")

train_data.info()  

X_train = train_data.drop('rating', axis=1) 
Y_train = train_data[['rating']] 


search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى",
              "\\",'\n', '\t','"','?','؟','!']
replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا",
               "","","","ي","",' ', ' ',' ',' ? ',' ؟ ', ' ! ']
def process_review(review):
    processed_review = re.sub(r"[^\w\s]", '', review)# accept all things accept any word,digit,white space and replace with empty string
    processed_review = re.sub(r"[a-zA-Z]", '', processed_review) # here I remove all english lettters
    for i in range(0, len(search)):
        processed_review = processed_review.replace(search[i], replace[i])
    processed_review = re.sub(r"\d+", '', processed_review)
    processed_review = re.sub(r"\n", '', processed_review)# here remove new line
    processed_review = re.sub(r"\s+", ' ', processed_review)# here remove white spaces
    processed_review = re.sub(r'(.)\1+', r'\1', processed_review)# here if there are any repeate in any character replace it with one occurance
    
    return processed_review.strip() # return review after precessed without any additional white space

processed_reviews=[]
ratings=[]
reviews=X_train.values
rating=Y_train.values

for i in range(len(reviews)):
    x=process_review(reviews[i][0])
    y=rating[i][0]
    processed_reviews.append(x)
    ratings.append(y)

def tokenize(sentence):
    words = []
    word = ""
    for char in sentence:
        if char == ' ':
            if word:
                words.append(word)
                word = ""
        else:
            word += char

    if word:
        words.append(word)

    return words

tokenized_reviews=[]

for i in range(len(processed_reviews)):
    t=tokenize(processed_reviews[i])
    tokenized_reviews.append(t)
    
# print(tokenized_reviews[500])

ar_stopwords = '''
أنفسنا مثل حيث ذلك بشكل لدى ألا عن إلي ب لنا وقالت فقط الذي الذى ا هذا غير أكثر اي أنا أنت ايضا اذا كيف وكل أو اكثر أي أن منه وكان وفي تلك إن سوف حين نفسها هكذا قبل حول منذ هنا عندما على ضمن لكن فيه عليه قليل صباحا لهم بان يكون بأن أما هناك مع فوق بسبب ما لا هذه و فيها ف ولم ل آخر ثانية انه من الان جدا به بن بعض حاليا بها هم أ كانت هي لها نحن تم أنفسهم ينبغي إلى فان وقد تحت' عند وجود الى فأن الي او قد خارج إنه اى مرة هؤلاء أنها إذا فهي فهى كل يمكن جميع أنفسكم فعل كان ثم لي الآن وقال فى في ديك لم لن له تكون الذين ليس التى التي أنه وان بعد حتى ان دون وأن لماذا يجري كلا إنها لك ضد وإن فهو انها منها أى لديه ولا بين خلال وما اما عليها بعيدا كما نفسي نحو هو نفسك نفسه انت ولن إضافي لقاء وكانت هى فما أيضا إلا معظم ومن إما الا بينما وهي وهو وهى
'''

ar_stopwords = tokenize(ar_stopwords)

# there are alot of stop words I can't write all of them can I use stop words in corpus of nltk
def remove_stop_words(text):
    filtered_words = [word for word in text if word not in ar_stopwords]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

reviews_after_stopWords=[]
for i in range(len(tokenized_reviews)):
    filtered_review = remove_stop_words(tokenized_reviews[i])
    reviews_after_stopWords.append(filtered_review)

from sklearn.model_selection import train_test_split
X_train, X_test, label_train, label_test = train_test_split(reviews_after_stopWords, ratings, test_size=0.2, random_state=42)

import pandas as pd

test_data=pd.read_csv("Datasets/test _no_label.csv")
x_test=test_data.drop('ID', axis=1) 
ID = test_data[['ID']]

X_train_t = []
label_train_t = []

for i in range(len(X_train)):
    if X_train[i] != "":
        X_train_t.append([i for i in X_train[i].split()])
        label_train_t.append(int(label_train[i]))
            
        
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_t)
vocab_size = len(tokenizer.word_index) + 1

x_train_sequences = tokenizer.texts_to_sequences(X_train_t)
max_len = max(len(seq) for seq in x_train_sequences)
x_train_padded = pad_sequences(x_train_sequences, maxlen=max_len, padding='post')
#max len = 295
# Label Encoding
label_train_t = np.array(label_train_t)
label_train_t_f = []
for i in label_train_t:
    if i == 1:
        res = [1,0,0]
    elif i == 0:
        res = [0,1,0]
    elif i == -1:
        res = [0,0,1]
    label_train_t_f.append(res)    
        

label_train_t_f = np.array(label_train_t_f)        
        

model = transformer.build_transformer_model(max_len, vocab_size)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_padded, label_train_t_f, epochs=3, batch_size=32)


X_test_t = []
label_test_t = []
for i in range(len(X_test)):
    if X_test[i] != "":
        X_test_t.append([i for i in X_test[i].split()])
        label_test_t.append(int(label_test[i]))

x_test_sequences = tokenizer.texts_to_sequences(X_test_t)

x_test_padded = pad_sequences(x_test_sequences, maxlen=max_len, padding='post')

# Label Encoding
label_test_t = np.array(label_test_t)
label_test_t_f = []
for i in label_test_t:
    if i == 1:
        res = [1,0,0]
    elif i == 0:
        res = [0,1,0]
    elif i == -1:
        res = [0,0,1]
    label_test_t_f.append(res)    
        
label_test_t_f = np.array(label_test_t_f) 

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test_padded, label_test_t_f)
print(f'Test Accuracy: {test_acc}')

predictions=model.predict(x_test_padded)
predictions=np.argmax(predictions,axis=1)

outputs=[]
for i in predictions:
    if i == 0 :
        x=1
    elif i==1:
        x=0
    elif i==2:
        x=-1
    outputs.append(x)

processed_reviews=[]
reviews=x_test.values
for i in range(len(reviews)):
    x=process_review(reviews[i][0])
    processed_reviews.append(x)

tokenized_reviews=[]

for i in range(len(processed_reviews)):
    t=tokenize(processed_reviews[i])
    tokenized_reviews.append(t)

test_reviews=[]
for i in range(len(tokenized_reviews)):
    filtered_review = remove_stop_words(tokenized_reviews[i])
    test_reviews.append(filtered_review)
    
    
x_test_t = []
for i in range(len(test_reviews)):
    if reviews[i] != "":
        x_test_t.append([i for i in test_reviews[i].split()])


x_test_sequences = tokenizer.texts_to_sequences(x_test_t)

x_test_padded = pad_sequences(x_test_sequences, maxlen=max_len, padding='post')

predictions_output_file=model.predict(x_test_padded)

predictions=np.argmax(predictions_output_file,axis=1)
#print(predictions)

outputs=[]
for i in predictions:
    if i == 0 :
        x=1
    elif i==1:
        x=0
    elif i==2:
        x=-1
    outputs.append(x)


with open('output_predition.csv',mode='w')as out:
    field=['ID','rating']
    writer=csv.DictWriter(out,fieldnames=field)
    writer.writerow({'ID':'ID','rating':'rating'})
    for i in range(1,1001,1):
        writer.writerow({'ID':i,'rating':outputs[i-1]})
        
print("File saved successfully")