from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

# Read text8 content
print("Loading content...")
with open("data/text8", "r") as file:
    text8_content = file.read()

# To sentences
print("Sentence tokenizing...")
text8_sentences = sent_tokenize(text8_content)

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
