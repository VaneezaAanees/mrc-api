from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
qa_pipeline = pipeline("question-answering")

class QARequest(BaseModel):
    context: str
    question: str

@app.post("/mrc")
def get_answer(request: QARequest):
    result = qa_pipeline({
        "context": request.context,
        "question": request.question
    })
    return {"answer": result["answer"]}