import pickle
import re
import nltk
from nltk.corpus import stopwords


class TextPreProcessor:
    def __init__(self, data_dir):
        # downloading nltk stuff
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

        # loading vectorizer from training
        with open(data_dir + "vectorizer.pkl", "rb") as vec_file:
            self.vectorizer = pickle.load(vec_file)

        self.a = ""

    def text_to_words(self, text):
        """Expects singular strings as input."""
        _ = self
        pattern = r'[^a-zA-Z\s]'
        # removing urls
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text = url_pattern.sub('', text)

        letters_only = re.sub(pattern, '', text)
        words = letters_only.lower().split()
        stop_words = set(stopwords.words('english'))

        lemmatizer = nltk.stem.WordNetLemmatizer()
        meaningful_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

        return " ".join(meaningful_words)

    def words_to_vec(self, words_str_or_str_array):
        if isinstance(words_str_or_str_array, str):
            return self.vectorizer.transform([words_str_or_str_array]).toarray()
        return self.vectorizer.transform(words_str_or_str_array).toarray()

    def pre_process_pipe(self, words):
        """Cleans text, then vectorizes."""
        if isinstance(words, list):
            wrds = [self.text_to_words(txt) for txt in words]
        else:
            wrds = self.text_to_words(words)

        return self.words_to_vec(wrds)
