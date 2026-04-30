import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from nltk import word_tokenize
try:
    nltk.download('punkt')
    nltk.download('wordnet')
except:
    pass

ref = "this is a test"
hyp = "this is a test"

print("CHRF:", sentence_chrf(ref.split(), hyp.split()))
print("METEOR:", meteor_score([word_tokenize(ref)], word_tokenize(hyp)))
