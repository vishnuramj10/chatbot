from flask import Flask, request, jsonify
from chatbot import load_model, load_pdf, create_chain, respond

app = Flask(__name__)

# Load the model and setup the question-answering chain
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
llm = load_model(model_name)
db = load_pdf("Sample RAG Questions.pdf")
qa_chain = create_chain(llm, db)

@app.route('/')
def query():
    query_text = request.args.get('query')
    if query_text:
        response = respond(qa_chain, query_text)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No query parameter provided'})

if __name__ == '__main__':
    app.run(debug=True)