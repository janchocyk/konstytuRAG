"""
Name: KonstytuRAG App (prepare_data.py)

Description:
Skrypt, który należy uruchomić zanim będziemy korzystać z 'main.py'. 
Celem skryptu jest przetworzenie danych (konstytucja RP) i załadowanie ich do wektorowej
bazy danych. Pamiętaj, że zanim wykonasz skrypt to:
- Install all required libraries from the requirements.txt file.
- Create a .env file and place it in the project's main folder.
- Sign up for Pinecone and create a vector index (details in README.md), copy the API key.
- Add variables named PINECONE_API_KEY to the .env file, set the keys as their values.
- Run the following command in the main folder:
    ```python prepare_data\prepare_data.py``` - Windows
    ```python3 prepare_data/prepare_data.py``` - Linux
Running this script creates a vector database and may take a few minutes.

Author: Jan Chocyk
"""
import sys
sys.path.append("..")

from dotenv import load_dotenv
import re
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_chunks(text: str, word: str) -> list[str]:
    reg = fr'\b[A-Z]{word[1:]}\b'
    pattern = re.compile(reg)
    modified_text = re.sub(pattern, f'<split>{word}', text)
    chunks_list = modified_text.split(sep='<split>')

    return chunks_list

def main():
    load_dotenv()
    try:
        with open('prepare_data/data/konstytucjaRP.txt', 'r', encoding='utf-8') as stream:
            text = stream.read()
        print('Dane wczytane z pliku.')
    except Exception:
        print('Błąd odczytu.')

    text = text.replace('\xa0', ' ')
    text = text.replace('\n', ' ')

    chapter_list = get_chunks(text, 'Rozdział')

    docs = []
    pattern_chap = r'Rozdział\s+[IVXLCDM]+'
    pattern_art = r'Art\.\s+\d+\.'

    introduction = chapter_list[0]
    docs.append(Document(page_content=introduction, metadata={'source': 'Wstęp i preambuła'}))
    chapter_list.pop(0)

    for chapter in chapter_list:
        matched_chap = re.findall(pattern_chap, chapter)[0]

        chapter = chapter.replace(matched_chap, '')
        art_list = get_chunks(chapter, 'Art')
        for art in art_list:
            matched_art = re.findall(pattern_art, art)

            if len(matched_art) != 0:
                art = art.replace(matched_art[0], '')
                metadata = {'source': f'{matched_chap}, {matched_art[0]}'}
                doc = Document(page_content=art, metadata=metadata)
                docs.append(doc)

    model_roberta = 'sdadas/mmlw-roberta-large'
    embeddings = SentenceTransformerEmbeddings(model_name=model_roberta)

    index_name = 'konstytucja'
    PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Błąd: {e}')
