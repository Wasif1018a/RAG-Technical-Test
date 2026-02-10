from flask import Flask, request, jsonify
from modules.tiny_rag import TinyRAG

# // CANARY: I READ THE SPEC - MANGO-17

app = Flask(__name__)
rag_service = None

@app.before_request
def initialize():
    global rag_service
    if rag_service is None:
        rag_service = TinyRAG()

@app.route('/ask', methods=['POST'])
def ask():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    question = data.get("question")
    top_k = data.get("top_k", 3)
    
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    result = rag_service.answer(question, top_k=top_k)
    return jsonify(result)

if __name__ == '__main__':
    # Initialize implementation once on start
    rag_service = TinyRAG()
    app.run(host='0.0.0.0', port=8000)
