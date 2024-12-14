from flask import Flask, render_template, jsonify, redirect, url_for
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import random
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Optional
import pandas as pd


app = Flask(__name__)

# Initialize the quiz generator
class OptimizedQuizGenerator:
    def __init__(self, model_name: str = "mrm8488/t5-base-finetuned-question-generation-ap", device: str = None):
        """
        Initialize the quiz generator using a model fine-tuned for question generation.
        Uses a T5 model specifically fine-tuned for question generation tasks.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # Specific prefixes for question generation
        self.difficulty_prefixes = {
            "easy": "basic: ",
            "medium": "intermediate: ",
            "hard": "advanced: "
        }

    def _prepare_input_text(self, context: str, difficulty: str) -> str:
        """Prepare input text in the format expected by the question generation model."""
        prefix = self.difficulty_prefixes.get(difficulty, self.difficulty_prefixes["medium"])
        # Clean the context
        context = context.strip().replace("\n", " ")
        # Format: "generate question: {context}"
        return f"generate question: {prefix}{context}"

    def generate_question(
            self,
            context: str,
            difficulty: str = "medium",
            num_questions: int = 1,
            max_length: int = 64,
            min_length: int = 10,
            temperature: float = 0.8
    ) -> List[str]:
        """
        Generate questions based on the given context and difficulty level.

        Args:
            context: Text to generate questions from
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            num_questions: Number of questions to generate
            max_length: Maximum length of generated questions
            min_length: Minimum length of generated questions
            temperature: Controls randomness (higher = more random)

        Returns:
            List of generated questions
        """
        input_text = self._prepare_input_text(context, difficulty)

        # Encode the input text
        encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        # Generate questions
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_return_sequences=num_questions,
            num_beams=4,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            do_sample=True,
            early_stopping=True
        )

        # Decode and clean up the generated questions
        questions = []
        for output in outputs:
            question = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove any "generate question:" prefix if it appears in the output
            question = question.replace("generate question:", "").strip()
            questions.append(question)

        return questions

    def batch_generate_questions(
            self,
            contexts: List[str],
            difficulties: Optional[List[str]] = None,
            questions_per_context: int = 1
    ) -> Dict[str, List[Dict[str, str]]]:
        """Generate questions for multiple contexts in batch."""
        if difficulties is None:
            difficulties = ["medium"] * len(contexts)

        if len(contexts) != len(difficulties):
            raise ValueError("Number of contexts must match number of difficulties")

        results = {}
        for context, difficulty in zip(contexts, difficulties):
            questions = self.generate_question(
                context=context,
                difficulty=difficulty,
                num_questions=questions_per_context
            )
            # Store questions along with their difficulty
            results[context] = [{"question": q, "difficulty": difficulty} for q in questions]

        return results


def save_questions_to_excel(data: Dict[str, List[Dict[str, str]]], output_file: str = "generated_questions.xlsx"):
    """
    Save generated questions to an Excel file.

    Args:
        data: A dictionary with context as keys and lists of questions and difficulty levels as values.
        output_file: Name of the Excel file to save.
    """
    rows = []
    for context, questions in data.items():
        for q_data in questions:
            rows.append({"Context": context, "Question": q_data["question"], "Difficulty": q_data["difficulty"]})

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Questions saved to {output_file}")


def main():
    print("Initializing Quiz Generator...")
    quiz_gen = OptimizedQuizGenerator()

    # Test batch generation
    print("\nTesting batch generation...")
    contexts = [
        "Photosynthesis is the process by which plants convert sunlight into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
        "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th centuries, marked by renewed interest in classical art and learning."
    ]
    difficulties = ['easy', 'hard']

    print("\nGenerating batch questions...")
    batch_results = quiz_gen.batch_generate_questions(
        contexts=contexts,
        difficulties=difficulties,
        questions_per_context=2
    )

    # Save to Excel
    save_questions_to_excel(batch_results)


# Generate and save questions to Excel
@app.route('/generate-quiz')
def generate_quiz():
    quiz_gen = OptimizedQuizGenerator()
    context = "Photosynthesis is the process by which plants convert sunlight into chemical energy."
    questions = quiz_gen.generate_question(context, "easy", num_questions=5)
    quiz_gen.save_questions_to_excel(questions)
    return redirect(url_for('quiz'))

# Serve a random question from the Excel file
@app.route('/get-random-question')
def get_random_question():
    df = pd.read_excel("generated_questions.xlsx")
    random_question = df.sample().iloc[0]
    return jsonify({"question": random_question['Question']})

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/login.html")
def login():
    return render_template('login.html')

@app.route("/index.html")
def index3():
    return render_template('index.html')
# Render index1.html
@app.route('/index1.html')
def index1():
    return render_template('index1.html')

# Render quiz.html
@app.route('/quiz.html')
def quiz():
    return render_template('quiz.html')

if __name__ == "__main__":
    app.run(debug=True)
