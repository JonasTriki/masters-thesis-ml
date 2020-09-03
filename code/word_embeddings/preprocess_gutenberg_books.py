from os.path import join as join_path

from .text_preprocessing_utils import preprocess_text
from .utils import (build_vocabulary, get_cached_download,
                   save_vocabulary_to_file)

# Constants
data_dir = "data"
gutenberg_books = [
    ("dracula", "http://www.gutenberg.org/cache/epub/345/pg345.txt"),
    ("alice_in_wonderland", "https://www.gutenberg.org/files/11/11-0.txt"),
]


def preprocess_data() -> None:
    """
    Preprocess data for training a Word2vec model.
    """
    # Preprocess all books
    for book_name, url in gutenberg_books:
        print(f"-- Processing {book_name}... --")

        # Fetch book content
        print("Fetching book content...")
        book_content = get_cached_download(book_name, data_dir, url)

        # Build vocabulary from text content
        print("Building vocabulary...")
        (
            book_word_dict,
            book_rev_word_dict,
            book_word_counts,
            book_word_noise_dict,
        ) = build_vocabulary(book_content, preprocess_text)

        # Save vocab to file
        print("Saving vocabulary to file...")
        vocab_filepath = join_path(data_dir, f"{book_name}_vocab.pickle")
        save_vocabulary_to_file(
            vocab_filepath,
            book_word_dict,
            book_rev_word_dict,
            book_word_counts,
            book_word_noise_dict,
        )

        print("Done!")


if __name__ == "__main__":
    preprocess_data()
