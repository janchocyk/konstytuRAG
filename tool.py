from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document


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
        retriever = self.vectorstore.as_retriver(k=2)

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

# przemyśleć zrócenie danych
# chcę docelowo ładną wiadomość do wyświetlenia: tex z llm'a + źródła
# więc chyba kierunkiem jest sklejenie tego w obrębie funkcji
    def get_answer(self, question, chat_history: list):
        ai_answer = self.rag_chain.invoke({"input": question, "chat_history": chat_history})

        return ai_answer['answer'], ai_answer['context'].metadata['source']