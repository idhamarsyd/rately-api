import re
import string
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

import math

import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# import contractions

DATA_SAMPLE = "The movie was great! The plot was engaging, and the acting was superb."

# Get the data from database
# Function of preprocessing
# Execute the data to be processed

class Preprocessing():
    def preprocessing_function(self, text):
        # text = self.Translation(text)
        text = self.Cleaning(text)
        text = self.CaseFolding(text)
        text = self.Tokenize(text)
        # text = self.StopwordRemoval(text)
        # text = self.Stemming(text)

        return text
    
    def Cleaning(self, text):

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove punction
        text = text.translate(str.maketrans("","",string.punctuation))

        # Remove whitespace leading & trailing
        text = text.strip()

        # Remove multiple whitespace into single whitespace
        text = re.sub('\s+',' ',text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        return text
    
    def Translation(self, text):
        translationEN = GoogleTranslator(source='auto', target='id').translate(text)
        return translationEN
    
    def CaseFolding(self, text):
        return text.lower()
    
    def Tokenize(self, text):
        token = text.translate(str.maketrans('','',string.punctuation)).lower()
        return token.split(' ')

    def StopwordRemoval(self, text):
        # # With NLTK (English Only)
        # stop_words = set(stopwords.words('english'))

        # filtered_sentence = [w for w in text if not w.lower() in stop_words]
        # filtered_sentence = []

        # for w in text:
        #     if w not in stop_words:
        #         filtered_sentence.append(w)

        # return filtered_sentence

        # Original
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        
        text = ' '.join(text)
        removed = stopword.remove(text)

        return removed.split(' ')
    
    def Stemming(self, text):
        # NLTK (English)
        # snow_stemmer = SnowballStemmer(language='english')

        # stemmed_sentence = []

        # for w in text:
        #     stemmed_text = snow_stemmer.stem(w)
        #     stemmed_sentence.append(stemmed_text)

        # return stemmed_sentence

        # Sastrawi (Indonesia)
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        text = ' '.join(text)
        removed = stemmer.stem(text)

        return removed.split(' ')

        # Create a cache for stemmed words
        stemmed_cache = {}

        def stem_word(word):
            if word not in stemmed_cache:
                stemmed_cache[word] = stemmer.stem(word)
            return stemmed_cache[word]

        stemmed_sentence = [stem_word(word) for word in text]
    
        return stemmed_sentence

    def ComputeTF(self, query, text, index):
        tokenText = text.split(' ')
        
        dataQuery = {}
        dataSentimen = {}

        xquery = [x[0] for x in query]

        sentimen = {
            'positif': 0,
            'negatif': 0
        }

        for i in query:
            dataQuery[i[0]] = 0
            dataSentimen[i[0]] = i[1]

        check = False
        for i in range(len(tokenText)):
            newToken = ''

            if check:
                check = False
                continue
            
            if i < len(tokenText)-1:
                newToken = tokenText[i]+" "+tokenText[i+1]
                if newToken in xquery:
                    if index==1:
                        print(newToken)
                    dataQuery[newToken] += 1
                    sentimen[dataSentimen[newToken]] += 1
                    check = True
                    i+=1

            if not check:
                newToken = tokenText[i]
                if newToken in xquery:
                    dataQuery[newToken] += 1
                    sentimen[dataSentimen[newToken]] += 1

            if index==1:
                print(newToken)

        return [dataQuery, sentimen]  

    def ComputeDF(self, query, tf, n):
        df = {}
        for i in query:
            df[i[0]] = 0

        doc_total = [0 for i in range(n)]

        for i in range(len(df)):
            for k in range(len(doc_total)):
                # print(tf[k][query[i][0]])
                if tf[k][query[i][0]]>0:
                    df[query[i][0]] += 1
                
        return df  
    
    def ComputeIDF(self, query, df, n):
        idf = {}
        # xquery = [x[0] for x in query]

        for i in query:
            if df[i[0]]!=0:
                idf[i[0]] = math.log10(n/int(df[i[0]]))
            else:
                idf[i[0]] = 0
                
        return idf
    
    def ComputeTFIDF(self, tf, idf):
        tfidf = []

        for i in range(len(tf)):
            # temp_W = {}
            temp_W = []
            for j in tf[i]:
                # print(i, j, tf[i][j], idf[j])
                # temp_W[j] = tf[i][j] * idf[j]
                temp_W.append(tf[i][j] * idf[j])
            
            # tfidf.append(temp_W)
            tfidf.append(temp_W)

        return tfidf