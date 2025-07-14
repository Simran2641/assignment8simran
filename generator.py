from transformers import pipeline

class Generator:
    def __init__(self):
        self.qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

    def generate_answer(self, query, context):
        prompt = f"Context: {context}\n\nQuestion: {query}"
        result = self.qa_pipeline(prompt, max_length=100)[0]['generated_text']
        return result
