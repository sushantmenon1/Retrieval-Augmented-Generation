from fastapi import FastAPI, UploadFile
from typing import Optional
from pydantic import BaseModel

from preprocessing import preprocess

import generate

app = FastAPI()
process = preprocess()

class inference_input(BaseModel):
    prompt: str
    hf_api_token: Optional[str] = "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"
    pinecone_index_name: Optional[str] = "retrieval-augmentation"
    modelid: Optional[str] = "google/flan-t5-xxl"

@app.post("/upload")
def upload(files: list[UploadFile],
           index_name: Optional[str] = "retrieval-augmentation",
           pinecone_api_key: Optional[str] = "2b414101-2b40-4c52-a9ed-92223cbff6f9",
           pinecone_env: Optional[str] = "gcp-starter"):
    process.push_to_pinecone(files,index_name, pinecone_api_key,pinecone_env)
    return {"Status":"Upload completed"}

@app.post("/inference")
def inference(input:inference_input):
    prompt, contexts = process.prepare_prompt(input.prompt, input.pinecone_index_name)
    pred = generate.query({"inputs": prompt}, input.modelid, input.hf_api_token)
    pred[0]['Metadata'] = contexts
    return pred