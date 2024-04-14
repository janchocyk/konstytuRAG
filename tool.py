from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
import tiktoken


contextualize_q_system_prompt = """Biorąc pod uwagę historię czatu i ostatnie pytanie użytkownika \
które może odnosić się do informacji z historii czatu, sformułuj samodzielne pytanie \
które będzie uwzględniało kontekst historii czatu. NIE odpowiadaj na pytanie, \
tylko przeformułuj je w razie potrzeby, a w przeciwnym razie zwróć je bez zmian.
"""

qa_system_prompt = """Jesteś asystentem, który odpowiada na pytania dotyczące Konstytucji Rzeczpospolitej Polskiej. \
Odpowiedz najuczciwiej i najdokładniej jak tylko umiesz, używając tylko i wyłącznie informacj zawartych w przekazanym kontekście. \
Jeżeli nie znajdziesz odpowiedzi na pytanie w kontekście odpowiedz: 'Niestety nie znam odpowiedzi'

{context}"""


class konstytuRAG():
    def __init__(self) -> None:
        self.embeddings = SentenceTransformerEmbeddings(model_name='sdadas/mmlw-roberta-large')
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        self.vectorstore = PineconeVectorStore.from_existing_index(index_name='konstytucja', embedding=self.embeddings)
        self.rag_chain = None
    
    def init_rag_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    def get_answer(self, question: str, chat_history: list):        
        ai_answer = self.rag_chain.invoke({'input': question, 'chat_history': chat_history})
        if ai_answer['answer'] == 'Niestety nie znam odpowiedzi.':
            source = 'Źródła: ----------' 
        else:
            source = 'Źródła: ' + ai_answer['context'][0].metadata['source'] + ', ' + ai_answer['context'][1].metadata['source']

        return ai_answer['answer'], source


def test_prompt(query, chat_history):
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        

    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    
    conteqt_prompt = contextualize_q_prompt.format(input=query, chat_history=chat_history)
    prompt_qa = qa_prompt.format(input=query, context='przykładowy kontekst...', chat_history=chat_history)
    print('Context prompt: ')
    print(conteqt_prompt)
    print('QA prompt: ')
    print(prompt_qa)

def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def customize_chat_history(chat_history: list):
    while True:
        if len(chat_history) < 3:
            break
        history_str = ''
        for mes in chat_history:
            history_str = history_str + mes['content']
        len_history = num_tokens_from_string(history_str, 'gpt-3.5-turbo')
        if len_history <2000:
            break
        if len(chat_history) >= 2:
            del chat_history[:2]   

    return chat_history
