import nltk
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class RNNModelManager:
        def __init__(self, sentences, labels):
            self.test_accuracy = None
            self.test_loss = None
            self.test_pred = None
            self.sentences = sentences
            self.labels = labels
            self.padded_sequences = None
            self.stop_words = None
            self.sequences = None
            self.train_inputs = None
            self.train_targets = None
            self.val_inputs = None
            self.val_targets = None
            self.test_inputs = None
            self.test_targets = None
            self.model = None
            self.vocab_size = None
            self.val_loss = None
            self.val_accuracy = None

        def preprocess_split(self, train_size, test_size):
            start = time.time()
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = nltk.corpus.stopwords.words('english')

            # removing stop words
            self.sentences = [self.remove_stop_words(sentence) for sentence in self.sentences]

            # tokenization
            tokenizer = tf.keras.preprocessing.text.Tokenizer()
            tokenizer.fit_on_texts(self.sentences)
            self.sequences = np.array(tokenizer.texts_to_sequences(self.sentences))

            # padding sequences to maximum length
            max_length = max([len(seq) for seq in self.sequences])
            self.padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(self.sequences, maxlen=max_length)
            self.vocab_size = len(tokenizer.word_index) + 1

            # stratify
            self.train_inputs, val_inputs, self.train_targets, val_targets = train_test_split(self.padded_sequences
                                                                        , self.labels
                                                                        , train_size=train_size
                                                                        , random_state=42)
            self.test_inputs, self.val_inputs, self.test_targets, self.val_targets = train_test_split(val_inputs
                                                                      , val_targets
                                                                      , test_size=test_size
                                                                      , random_state=42)
            print("--- preprocessing done in %s s ---" % (time.time() - start))

        def remove_stop_words(self, sentence):
            return [word for word in sentence if word not in self.stop_words]

        def compile_fit_model(self, model, epochs):
            start = time.time()
            self.model = model
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.model.fit(self.train_inputs, self.train_targets, epochs=epochs) # , validation_split=0.2, shuffle=True
            print("--- model training done in %s s ---" % (time.time() - start))

        def evaluate(self):
            start = time.time()
            self.val_loss, self.val_accuracy = self.model.evaluate(self.val_inputs, self.val_targets)
            print("--- evaluation done in %s seconds ---" % (time.time() - start))

        def predict_report(self):
            start = time.time()
            self.test_pred = self.model.predict(self.test_inputs)
            # false positive rate and true positive rate
            fpr, tpr, thresholds = roc_curve(self.test_targets, self.test_pred)
            roc_auc = auc(fpr, tpr)

            # print("False-Positive Rate: ", fpr)
            # print("True-Positive Rate: ", tpr)
            # print("Thresholds: ", thresholds)
            print("Validation Loss: {:.4f}".format(self.val_loss))
            print("Validation Accuracy: {:.4f}".format(self.val_accuracy))
            print("Test AUC: ", roc_auc)

            # ROC (AUC)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()

            print("--- prediction done in %s seconds ---" % (time.time() - start))
