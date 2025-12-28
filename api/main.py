from fastapi import FastAPI, HTTPException

from api.models import SceneRequest, SceneResponse
from api.generation import write_scene

app = FastAPI()

@app.get("/")
def index():
    return "Hello"

@app.post("/generate")
def request_scene(scene_request: SceneRequest) -> SceneResponse:
    print("the request payload: {}".format(scene_request))
    
    scene = write_scene(scene_request)
    
    return scene

@app.post("/generate/test")
def request_test_scene(scene_request: SceneRequest) -> SceneResponse:
    print("the request payload: {}".format(scene_request))
    
    scene = write_scene(scene_request, test_response=True)
    
    return scene

