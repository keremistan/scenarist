from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class ScoreCard(BaseModel):
    coherence: int = Field(description="does the text make sense", ge=1, le=5)
    style_adherence: int = Field(description="does the text adhere the style reference", ge=1, le=5)
    reasoning: str = Field(description="a short explanation of why you graded as you did")

def judge(generated_scene: str,style_reference: str, user_intent: str):
    
    the_judge = ChatOllama(model='gpt-oss:20b', reasoning='high'
                           ).with_structured_output(ScoreCard)
    
    system_prompt = {
        "role": "system",
        "content": "You are a film critic. Compare this GENERATED SCENE against these STYLE REFERENCE and the USER INTENT. Grade it."
    }
    
    input_prompt = {
        "role": "user",
        "content": "USER INTENT: {}\nSTYLE REFERENCE: {}\nGENERATED SCENE: {}\n".format(user_intent, style_reference, generated_scene)
    }
    
    response = the_judge.invoke([
        system_prompt,
        input_prompt
    ])
    
    print("the judge's response:\n{}\n\n\n".format(response))
    
    return response
    
