# konstytuRAG
### Description
A web application that implements a chat interface where you can converse with a chatbot about the Polish Constitution. 
The RAG building technique was used, where the data on which the LLM bases its responses is the content of the constitution. 
The chatbot additionally provides sources it relied on to build its response.

### Uruchomienie:
1. Locally
- Install all required libraries from the requirements.txt file
    ```pip install -r requirements.txt```
- Create a .env file and place it in the project's main folder
- Sign up for Pinecone ([link](https://www.pinecone.io/)) and create a vector index (name='konstytucja'), copy the API key
- Sign up for the OpenAI ([link](https://openai.com/)) service and copy the API key
- Add variables named PINECONE_API_KEY and OPENAI_API_KEY to the .env file, set the keys as their values
- Upload data to database:
  Run: ```python prepare_data\prepare_data.py```
  Running this script creates a vector database and may take a few minutes.
- Start app:
  Run: ```streamlit run app\main.py --server.port 8080```
  After running, the default browser window will open with the application.
  
2. Via Docker
- Create any folder and place the downloaded DOCKERFILE from the repository in it
- Prepare the ```.env``` file, as described above, and place it in the same folder
- Run the commands:
    ```docker build -t konstyturag .```
    ```docker run -d -p 8080:8080 konstyturag```
  A Docker image will be created, and a container will be launched based on it.

### Learning
If you want to learn more and follow step-by-step the building of the project, 
take a look at the prepare_data.ipynb notebook.

### Technologies:
- Streamlit
- Langchain
- OpenAI
- Transformers

Author: Jan Chocyk
Version: 1.0
