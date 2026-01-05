from app.tools.loader import ScreenplayLoader
from app.tools.scene_analyzer import analyze_scene, SceneAnalysis
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import Optional
from langfuse import observe

from app.tools.traced_embeddings import TracedEmbeddings


@observe(name="ingestion" ,as_type="span")
def ingest(
    screenplay_address: str, 
    scenes_to_index: Optional[int] = None, 
    scenes_from_index: Optional[int] = None
    ):
    
    screenplay_loader = ScreenplayLoader(screenplay_address)
    scene_docs: list[Document] = []

    screenplays_vector_store_collection = Chroma(
        collection_name="screenplays", 
        embedding_function=TracedEmbeddings(),
        persist_directory='app/chroma')

    # extract text from pdf
    for i, current_scene in enumerate(screenplay_loader.lazy_load()):
        if scenes_from_index and i < scenes_from_index:
            continue
        
        scene_docs.append(current_scene)
        
        if scenes_to_index and i >= scenes_to_index:
            break

    # analyse each scene and insert it into vector store
    for current_scene_doc in scene_docs: # TODO: analysis takes a lot of time - might be parallelized
        scene_analysis = analyze_scene(current_scene_doc.page_content) 
        
        if scene_analysis:

            original_text = current_scene_doc.page_content

            # insert the analysis into the original doc
            current_scene_doc.page_content = "HAPPENING: {} \nSUBTEXT_LEVEL_HAPPENING: {} \nREADER_REACTION: {}".format(
                scene_analysis.happening,
                scene_analysis.subtext_level_happening,
                scene_analysis.reader_reaction
            )
            
            current_scene_doc.metadata['original_text'] = original_text
        
            screenplays_vector_store_collection.add_documents(
                [current_scene_doc]
            )

    return scene_docs

    
if __name__ == "__main__":
    # screenplay_address = 'screenplays/Zootopia!.pdf'

    # for screenplay_address in [
    #     # 'screenplays/Up!.pdf',
    #     # 'screenplays/Wall-e!.pdf',
    #     'screenplays/the-substance-2024!.pdf',
    #     'screenplays/The silence of the lambs!.pdf',
    #     ]:
    #     ingest(screenplay_address)
    
    ingest('screenplays/the-substance-2024!.pdf', scenes_from_index=75)