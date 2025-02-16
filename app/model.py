import chromadb
import logging
import sys

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


global query_engine
query_engine = None

def init_llm():
    """ Créez un nouveau modèle en analysant et 
    en validant les données d’entrée à partir d’arguments de mots-clés."""
    llm = Ollama(model="llama3.2", request_timeout=3000.0)
    
    """ Créez un nouveau modèle en analysant et en validant les données 
    d’entrée à partir d’arguments de mots-clés. La classe de base HuggingFaceEmbedding 
    est un wrapper générique autour de n'importe quel modèle HuggingFace pour les intégrations. 
    Tous les modèles d'intégration sur Hugging Face devraient fonctionner. Vous pouvez vous référer au classement des intégrations pour plus de recommandations."""
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model


def init_index(embed_model):
    """ Le SimpleDirectoryReader est le connecteur de données le plus couramment utilisé qui fonctionne tout simplement. Transmettez simplement un répertoire d'entrée ou une liste de fichiers."""
    reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
    
    """ Chargez les données à partir du répertoire d'entrée ou de la liste de fichiers."""
    documents = reader.load_data()

    logging.info("index creating with `%d` documents", len(documents))

    """Crée une instance en mémoire de Chroma. Cela est utile pour les tests, les démonstrations et les applications où la persistance n'est pas nécessaire."""
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("iollama")

    """ChromaVectorStore est une implémentation concrète de l'interface VectorStore. Elle est responsable de l'ajout, de la suppression et de la recherche de documents en fonction de leur similarité avec une requête, en utilisant ChromaApi et EmbeddingClient pour les calculs d'intégration"""
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    """Le conteneur de contexte de stockage est un conteneur utilitaire pour stocker des nœuds, des index et des vecteurs. Il contient les éléments suivants :
    docstore : BaseDocumentStore
    index_store : BaseIndexStore
    vector_store : BasePydanticVectorStore
    graph_store : GraphStore
    property_graph_store : PropertyGraphStore (initialisé paresseusement)"""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # use this to set custom chunk size and splitting
    # https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
    """Les Vector Stores sont un composant clé de la génération augmentée par récupération (RAG) et utilisé dans presque toutes les applications que vous créez à l'aide de LlamaIndex, directement ou indirectement. Les magasins vectoriels acceptent une liste d'objets Node et créent un index à partir de ceux-ci."""
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    return index


def init_query_engine(index):
    global query_engine

    """Le modèle qui est utilisé pour intégrer des requêtes et des documents dans des vecteurs. Il est utilisé pour la recherche de documents similaires dans un VectorStore."""
    template = (
        "Imaginez que vous êtes un expert avancé dans l'administration de bases de données pour une école française, avec accès à tous les documents actuels et pertinents. "
        "Votre objectif est de fournir des réponses perspicaces, précises et concises aux questions dans ce domaine.\n\n"
        "Voici un peu de contexte lié à la requête:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Compte tenu des informations ci-dessus, veuillez répondre à la demande suivante avec des références détaillées et des explications claires, en utilisant des termes techniques et des concepts pertinents, et en incluant des exemples si nécessaire. "
        "Question: {query_str}\n\n"
        "Répondez succinctement, en commençant par la phrase « D'après les données dont je dispose », et assurez-vous que votre réponse est compréhensible pour quelqu'un qui n'a pas un niveau technique élevé."
    )
    qa_template = PromptTemplate(template)
    # similarity_top_k = Nombre de documents pertinents à récupérer
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

    return query_engine


def chat(input_question, user):
    global query_engine

    response = query_engine.query(input_question)
    logging.info("got response from llm - %s", response)

    return response.response


def chat_cmd():
    global query_engine

    while True:
        input_question = input("Enter your question (or 'exit' to quit): ")
        if input_question.lower() == 'exit':
            break

        response = query_engine.query(input_question)
        logging.info("got response from llm - %s", response)


if __name__ == '__main__':
    init_llm()
    index = init_index(Settings.embed_model)
    init_query_engine(index)
    chat_cmd()