# TODO: Integrate the parsing of wikiextractor match Word2vec file format: lower case words separated by space, new line = new sentence.

# To the following to download enwiki:
# 1. Set wiki=enwiki, dump_time=20200901
# 2. Download https://dumps.wikimedia.org/{wiki}/{dump_time}/{wiki}-{dump_time}-pages-articles-multistream.xml.bz2 to raw_data
# 3. python -m wikiextractor.WikiExtractor -cb 250K -o raw_data/enwiki_extracted raw_data/{wiki}-{dump_time}-pages-articles-multistream.xml.bz

import argparse
import bz2
import logging

from datetime import datetime
from os import listdir
from os.path import isfile, join, isdir
from bs4 import BeautifulSoup
from tqdm import tqdm
from multiprocessing.dummy import Pool

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )

FORMAT = '%(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_all_files_recursively(root):
    files = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
    dirs = [d for d in listdir(root) if isdir(join(root, d))]
    for d in dirs:
        files_in_d = get_all_files_recursively(join(root, d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(f))
    return files


def process_wiki_file(file):
    with bz2.open(file, 'rt', encoding="utf8") as bz2_file:
        #logger.info(f"Reading/Writing file ---> {file}")

        # Extract text between <doc> xml tags
        soup = BeautifulSoup(bz2_file.read(), "lxml")
        docs = soup.find_all("doc")
        result = ""
        for doc in docs:
            result += f"{doc.text}\n"
        return result

def bzip_decompress(list_of_files, output_file):
    start_time = datetime.now()
    print(rawgencount(output_file))
    with Pool() as pool:
        with open(f'{output_file}', 'w', encoding="utf8") as output_file:
            for result in tqdm(pool.imap_unordered(process_wiki_file, list_of_files), total=len(list_of_files)):
                output_file.writelines(result)
                output_file.write('\n')
    stop_time = datetime.now()
    print(f"Total time taken to write out {len(list_of_files)} files = {(stop_time - start_time).total_seconds()}")


def main():
    parser = argparse.ArgumentParser(description="Input fields")
    parser.add_argument("-r", required=True)
    parser.add_argument("-n", required=False)
    parser.add_argument("-o", required=True)
    args = parser.parse_args()

    all_files = get_all_files_recursively(args.r)
    if args.n == None:
        n = len(all_files)
    else:
        n = int(args.n)
    bzip_decompress(all_files[:n], args.o)


if __name__ == "__main__":
    main()