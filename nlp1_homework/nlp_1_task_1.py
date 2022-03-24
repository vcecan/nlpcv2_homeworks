import spacy
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nlp = spacy.load('en_core_web_sm')
#-----------------task 1---------------
with open ("task1.txt","r") as txt:
    text = txt.read()
    txt.close()
#-------------------task 2----------------------------
count_prop=0
count_cuv=[]
counter=0
doc=nlp(text)

clean_doc = [token for token in doc if not token.is_stop and not token.is_punct \
             and not token.is_space ]
tokens=[]
clean_tokens = []
for token in doc:
    tokens.append(token.text)
for token in doc:
    if not token.is_stop and not token.is_punct and not token.is_space:
        clean_tokens.append(token.text)

uniq_count =0 #count pentru cuvinte unice
counts = Counter(tokens)
for sent in doc.sents:
     count_prop = count_prop + 1 #count de propozitii
     count_cuv.append(counter) # list de counts de cuvinte intr-o propozitie
     counter=0 # count pentru cuvinte pe propozitie, la fiecare propozitie se reintoarce la 0
     for token in sent:
         if not token.is_space and not token.is_punct :
            counter=counter+1
         print(token)
         if counts[token.text]==1:
             uniq_count =uniq_count+ 1 # numarul de cuvinte unice

del count_cuv[0]
media = sum(count_cuv)/len(count_cuv)
print(f'numarul de propozitii {count_prop}')
print(count_cuv)
print(int(media))
print(f'numar de tokenuri unice {uniq_count}')

#-----------------------------task 3---------------------------
lemma_token = []
for token in clean_doc[:50]:
        lemma_token.append(token.lemma_)
token_data = {
    "original_token": clean_tokens[:50],
    "Lemma": lemma_token[:50]
}
df = pd.DataFrame(token_data)
print(df)
#---------------------------task 4------------------------
def spacy_tokenizer(sentence):
    mytokens = nlp(sentence)

    clean_doc = [token.lemma for token in mytokens if not token.is_stop \
                 and not token.is_space and not token.is_punct]
    return clean_doc
bow_vectorizer = CountVectorizer(tokenizer = spacy_tokenizer)
baza_de_date = ["The pound  die requires no press, but instead, uses a mallet",
                "It is somewhat lower in cost because you do not need to purchase a press, but it is much slower to  use and doesn't produce jacketed bullets"]
bag_of_words=bow_vectorizer.fit_transform(baza_de_date).toarray()
print(bag_of_words)
#------------------------------task 4.2------------------------------
tfid = TfidfVectorizer(tokenizer = spacy_tokenizer)
result_tfid=tfid.fit_transform(baza_de_date).toarray()
print(result_tfid)

#--------------------------------task 5--------------------------------
print("verbe",[token for token in doc if token.pos_=="VERB"])
print("substantive",[token for token in doc if token.pos_=="NOUN"])
print("adjective",[token for token in doc if token.pos_=="ADJ"])
print("adverbe",[token for token in doc if token.pos_=="ADP"])
print("auxiliar",[token for token in doc if token.pos_=="AUX"])
print("determinative",[token for token in doc if token.pos_=="DET"])
#----------------------------------task 6 --------------------------
from spacy import displacy
task6 = displacy.render (doc,style='ent',jupyter=False )
print (task6)








