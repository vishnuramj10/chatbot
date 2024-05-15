import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from fastai.vision.all import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import urllib.request
from chatbot import load_model, load_pdf, create_chain, respond

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])

# Load the model and setup the question-answering chain
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
llm = load_model(model_name)
db = load_pdf("Sample RAG Questions.pdf")
qa_chain = create_chain(llm, db)

@app.route('/query', methods=['POST'])
async def analyze(request):
    query = await request.form()
    response = respond(qa_chain, query_text)
    return JSONResponse({'result': str(response)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5500, log_level="info")