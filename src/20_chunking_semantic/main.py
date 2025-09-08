from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from supabase import create_client, Client
import uuid
import os
import dotenv
dotenv.load_dotenv(".env")

# Connexión a la BD
def connect_to_db(url, key) -> Client:
    try:
        supabase: Client = create_client(url, key)
        print("Conexión exitosa.")
        return supabase
    
    except Exception as e:
        print("Error e la conexión: ", e)


def load_pdf(path: str):
    loader = PyPDFLoader(path)
    docs_raw = loader.load()
    print(f"PDF contiene {len(docs_raw)} páginas.")
    return docs_raw

# Dividir el PDF en chunks
docs_raw = load_pdf("files/Documento de servicio_1.1.pdf")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
text_spliter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

# Dividir texto con SemanticChunker
print("Dividiendo el texto semanticamente.")
texto_completo = "".join([doc.page_content for doc in docs_raw])
docs_chunks = text_spliter.create_documents([texto_completo])
print(f"El documento se ha dividido en {len(docs_chunks)} chunks semanticos.")

# Cargar chunks a Supabase
supabase = connect_to_db(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
for i, chunk in enumerate(docs_chunks):
    contenido = chunk.page_content
    embedding = embeddings.embed_query(contenido)

    try:
        supabase.from_('documentos').insert({
            "id": str(uuid.uuid4()),
            "texto": contenido,            
            "embedding": embedding,            
        }).execute()
        print(f"Chunk {i+1}. Cargado a Supabase.")
    except Exception as e:
        print(f"Error cargando chunk {i + 1} a Supabase: {e}")