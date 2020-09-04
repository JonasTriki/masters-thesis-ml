from os import makedirs

from text_preprocessing_utils import preprocess_text
from utils import get_cached_download_text_file

# Constants
raw_data_dir = "raw_data"
makedirs(raw_data_dir, exist_ok=True)
gutenberg_books = [
    ("dracula", "http://www.gutenberg.org/cache/epub/345/pg345.txt"),
    ("alice_in_wonderland", "https://www.gutenberg.org/files/11/11-0.txt"),
]
# TODO: ^ Indicate the starting and ending part of the book.


def preprocess_data() -> None:
    """
    Preprocess data for training a Word2vec model.
    """
    # Preprocess all books
    for book_name, url in gutenberg_books:
        print(f"-- Processing {book_name}... --")

        # Fetch book content
        print("Fetching book content...")
        book_content = get_cached_download_text_file(
            url, raw_data_dir, f"{book_name}.txt"
        )

        # TODO: Preprocess the content of the book

        print("Done!")


if __name__ == "__main__":
    preprocess_data()
