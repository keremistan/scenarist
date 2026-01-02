from fastapi import FastAPI

from app.api.models import SceneRequest, SceneResponse
from app.api.generation import write_scene
from app.tools.retrieve import SceneRetriever
import os
app = FastAPI()

@app.get("/")
def index():
    return "Hello"

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

