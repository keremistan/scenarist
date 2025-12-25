from loader import ScreenplayLoader
from scene_analyzer import analyze_scene, SceneAnalysis
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import Optional
from langchain_ollama.embeddings import OllamaEmbeddings

HOW_MANY_SCENES: Optional[int] = 5
screenplay_address = 'metropolis.pdf'

screenplay_loader = ScreenplayLoader(screenplay_address)

scene_docs: list[Document] = []
scene_analysises: list[SceneAnalysis] = []

for i, current_scene in enumerate(screenplay_loader.lazy_load()):
    scene_docs.append(current_scene)
    if HOW_MANY_SCENES and i >= HOW_MANY_SCENES:
        break

for current_scene_doc in scene_docs:
    scene_analysis = analyze_scene(current_scene_doc.page_content)
    if scene_analysis:
        scene_analysises.append(scene_analysis)
        
assert len(scene_docs) == len(scene_analysises)

# insert the analysis into the original doc
for i, scene_doc in enumerate(scene_docs):
    relevant_analysis = scene_analysises[i]
    
    scene_doc.metadata['happening'] = relevant_analysis.happening
    scene_doc.metadata['subtext_level_happening'] = relevant_analysis.subtext_level_happening
    scene_doc.metadata['reader_reaction'] = relevant_analysis.reader_reaction
    

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
screenplays_vector_store_collection = Chroma(
    collection_name="screenplays", 
    embedding_function=embedding_model, 
    persist_directory='./chroma')


screenplays_vector_store_collection.add_documents(
    documents=scene_docs
)

# show content of the chroma collection
res = screenplays_vector_store_collection.get()
print(res)