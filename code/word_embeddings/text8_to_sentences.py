import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

# Read text8 content
print("Loading content...")
with open("data/text8", "r") as file:
    text8_content = file.read()

# To sentences
print("Sentence tokenizing...")
text8_sentences = sent_tokenize(text8_content)

# Merge sentences with less than N words into previous sentence
# new_sentences = []
# for i in tqdm(range(0, len(text8_sentences) - 1, 2)):
#     j = i + 1

#     # Tokenize sentences
#     sent_words_i = word_tokenize(text8_sentences[i])[:-1]
#     sent_words_j = word_tokenize(text8_sentences[j])[:-1]

#     if len(sent_words_i) > 0 and len(sent_words_j) > 0:

#         # Ensure than sentence #i is longer than sentence
#         if len(sent_words_i) > len(sent_words_j) and len(sent_words_j) <= 3:
#             sent = " ".join(sent_words_i + sent_words_j)
#             new_sentences.append(sent)
#         else:
#             new_sentences.append(" ".join(sent_words_i))
#             new_sentences.append(" ".join(sent_words_j))
#     else:
#         if len(sent_words_i) > 0:
#             new_sentences.append(" ".join(sent_words_i))
#         if len(sent_words_j) > 0:
#             new_sentences.append(" ".join(sent_words_j))

# Filter out sentences that have less than N words in them
min_sent_word_count = 5
text8_sentences = [
    # -1 and [:-1] because "." will get tokenized as a word
    " ".join(word_tokenize(sent)[:-1])
    for sent in tqdm(text8_sentences)
    if len(word_tokenize(sent)) - 1 > min_sent_word_count
]

# Save to file
print("Saving to file...")
with open("data/text8_sents", "w") as file:
    for sent in text8_sentences:
        file.write(f"{sent}\n")

print("Done!")
