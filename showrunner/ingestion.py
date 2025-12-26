from loader import ScreenplayLoader
from scene_analyzer import analyze_scene, SceneAnalysis
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import Optional
from langchain_ollama.embeddings import OllamaEmbeddings

HOW_MANY_SCENES: Optional[int] = None
SCENES_FROM_INDEX: Optional[int] = None
screenplay_address = 'showrunner/screenplays/metropolis.pdf'

screenplay_loader = ScreenplayLoader(screenplay_address)

scene_docs: list[Document] = []
scene_analysises: list[SceneAnalysis] = []

for i, current_scene in enumerate(screenplay_loader.lazy_load()):
    if SCENES_FROM_INDEX and i < SCENES_FROM_INDEX:
        continue
    
    scene_docs.append(current_scene)
    
    if HOW_MANY_SCENES and i >= HOW_MANY_SCENES:
        break

for current_scene_doc in scene_docs: # TODO: this takes a lot of time - might be parallelized
    scene_analysis = analyze_scene(current_scene_doc.page_content) 
    if scene_analysis:
        scene_analysises.append(scene_analysis)
        
assert len(scene_docs) == len(scene_analysises)

# insert the analysis into the original doc
for i, scene_doc in enumerate(scene_docs):
    original_text = scene_doc.page_content
    
    relevant_analysis = scene_analysises[i]
    scene_doc.page_content = "HAPPENING: {} \nSUBTEXT_LEVEL_HAPPENING: {} \nREADER_REACTION: {}".format(
        relevant_analysis.happening,
        relevant_analysis.subtext_level_happening,
        relevant_analysis.reader_reaction
    )
    
    scene_doc.metadata['original_text'] = original_text
    

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