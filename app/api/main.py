from fastapi import FastAPI

from app.api.models import SceneRequest, SceneResponse
from app.api.generation import write_scene
from app.tools.retrieve import SceneRetriever
import os
app = FastAPI()

@app.get("/")
def index():
    return "Hello "

@app.get("/chroma")
def get_scenes():
    # todo: enable access to this endpoint only in dev
    # to test the chroma and ollama (embedding) connection
    print("in main.py the ollama base -> http://{}:11434".format(os.getenv("OLLAMA_DOCKER_SERVICE")))
    topics = ['heated argument', 'love confession', 'emotional conflict', 'tension', 'reconciliation attempt', 'intense emotion', 'contradictory feelings', 'fragile affection', 'heartbreak', 'understanding', 'conflict resolution', 'passionate dispute']

    scene_retriever = SceneRetriever()
    scenes = scene_retriever.query(", ".join(topics))

    return "scenes: {}".format(scenes)

@app.post("/generate")
def request_scene(scene_request: SceneRequest) -> SceneResponse:
    print("the request payload is: {}".format(scene_request))
    
    scene = write_scene(scene_request)
    
    return scene

@app.post("/generate/test")
def request_test_scene(scene_request: SceneRequest) -> SceneResponse:
    print("the request payload: {}".format(scene_request))
    
    scene = write_scene(scene_request, test_response=True)
    
    return scene

