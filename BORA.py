import os
from datasets import load_dataset
from openai import OpenAI
import json
import logging
from logging.handlers import RotatingFileHandler
import time
from functools import wraps
import pickle

# Logging configuration
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "dataset_processing.log")
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    
    return root_logger

logger = setup_logging()

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL","https://api.openai.com")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

def exponential_backoff_retry(max_retries=None, initial_delay=1, max_delay=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            retry_count = 0
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if max_retries is not None and retry_count > max_retries:
                        raise
                    
                    wait_time = min(delay * (2 ** (retry_count - 1)), max_delay)
                    logger.warning(f"Error in {func.__name__}: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        return wrapper
    return decorator

class ProcessingState:
    def __init__(self, state_file="BORA_processing_state.pkl"):
        self.state_file = state_file
        self.last_processed_idx = None
        self.processed_chunks = set()
        self.load_state()
    
    def save_state(self):
        with open(self.state_file, "wb") as f:
            pickle.dump({
                "last_processed_idx": self.last_processed_idx,
                "processed_chunks": self.processed_chunks
            }, f)
    
    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "rb") as f:
                    state = pickle.load(f)
                    self.last_processed_idx = state["last_processed_idx"]
                    self.processed_chunks = state["processed_chunks"]
                logger.info(f"Resumed from index {self.last_processed_idx}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")

def chunk_text(text, max_chars=2000):
    """Split text into chunks of approximately max_chars characters at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    sentences = text.replace("\n", " ").split(". ")
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# JSON Schema definitions
QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "preguntas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pregunta": {
                        "type": "string",
                        "description": "Una pregunta relevante basada en el contexto proporcionado"
                    },
                    "razonamiento": {
                        "type": "string",
                        "description": "Explicación del enfoque de la pregunta"
                    }
                },
                "required": ["pregunta", "razonamiento"]
            },
            "minItems": 2,
            "maxItems": 2
        }
    },
    "required": ["preguntas"]
}

CONTENT_EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valuable": {
            "type": "boolean",
            "description": "Whether the content is valuable for the dataset"
        },
        "reason": {
            "type": "string",
            "description": "Explanation for the evaluation decision"
        }
    },
    "required": ["is_valuable", "reason"]
}

QUESTION_EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_question": {
            "type": ["integer", "null"],
            "enum": [1, 2, None],
            "description": "Which question was selected (1, 2, or null if none)"
        },
        "reason": {
            "type": "string",
            "description": "Explanation for the selection decision"
        }
    },
    "required": ["selected_question", "reason"]
}

@exponential_backoff_retry(max_retries=None)
def generate_question(context):
    """Generate two different questions based on the context."""
    logger.debug("Generating questions for context of length %d", len(context))
    prompt = f"""Actúa como un abogado buscando información legal.
            Genera DOS preguntas diferentes que:
            1. Sean autocontenidas y no hagan referencia al texto o contexto
            2. Usen lenguaje cotidiano mezclado con términos legales relevantes
            3. Representen dudas reales sobre el tema principal del texto
            4. Sean claras y específicas, evitando ambigüedades o generalidades
            5. Eviten frases como "según el texto", "de acuerdo a la ley", "en este caso", etc.
            6. Sean distintas entre sí en enfoque o aspecto del tema que abordan
            
            Las preguntas deben ser diferentes pero igualmente válidas.
            Incluye un razonamiento breve que explique el enfoque de cada pregunta.
            
            Fragmento a analizar: {context}
            
            Responde con un JSON que contenga el array de preguntas y sus razonamientos"""

    try:
        response = client.chat.completions.create(
            model="",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={ "type": "json_object", "json_schema": QUESTIONS_SCHEMA }
        )
        
        content = response.choices[0].message.content
        logger.info(f"LLM Response: {content}")
        
        result = json.loads(content)
        return result["preguntas"]
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return [{"pregunta": "¿Qué información relevante contiene este texto?", "razonamiento": "Pregunta por defecto"}]

@exponential_backoff_retry(max_retries=None)
def evaluate_content_quality(context, questions=None):
    """Evaluate content and questions for RAG dataset suitability."""
    
    if questions is None:
        # Context-only evaluation prompt
        context_prompt = """Actúa como un evaluador experto en calidad de datos para sistemas RAG.
        Analiza si este fragmento de texto legal es adecuado como fuente de información según estos criterios:

        Criterios de evaluación del contenido:
        1. Sustancia legal: contiene normas, procedimientos o conceptos legales relevantes
        2. Aplicabilidad práctica: aborda situaciones o consultas realistas
        3. Potencial de consulta: podría responder a consultas específicas
        
        Contexto a evaluar: {context}
        
        Responde con un JSON que contenga:
        {{
            "is_valuable": true/false,
            "reason": "Explicación de la decisión"
        }}"""

        try:
            response = client.chat.completions.create(
                model="",
                messages=[{
                    "role": "user", 
                    "content": context_prompt.format(context=context)
                }],
                temperature=0.3,
                response_format={ "type": "json_object", "json_schema": CONTENT_EVALUATION_SCHEMA }
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("is_valuable", False), result.get("reason", "No reason provided")
        except Exception as e:
            logger.error(f"Error evaluating quality: {str(e)}")
            return False, "Error in evaluation"
    else:
        question_eval_prompt = """Actúa como un evaluador experto en calidad de datos para sistemas RAG.
        Analiza estas dos preguntas y selecciona la más adecuada para entrenar un sistema RAG, o ninguna si ambas son inadecuadas.

        Criterios de evaluación:
        1. Naturalidad: refleja consultas reales de usuarios
        2. Utilidad para embeddings: ayuda al modelo a aprender asociaciones semánticas
        3. Relevancia contextual: se relaciona directamente con el contenido principal
        4. Especificidad: es precisa y bien enfocada
        5. Independencia: no hace referencia al texto fuente

        Contexto: {context}

        Pregunta 1: {q1}
        Razonamiento 1: {r1}

        Pregunta 2: {q2}
        Razonamiento 2: {r2}

        Responde con un JSON:
        {{
            "selected_question": null/1/2,
            "reason": "Explicación de la selección"
        }}"""

        try:
            response = client.chat.completions.create(
                model="",
                messages=[{
                    "role": "user", 
                    "content": question_eval_prompt.format(
                        context=context,
                        q1=questions[0]["pregunta"],
                        r1=questions[0]["razonamiento"],
                        q2=questions[1]["pregunta"],
                        r2=questions[1]["razonamiento"]
                    )
                }],
                temperature=0.3,
                response_format={ "type": "json_object", "json_schema": QUESTION_EVALUATION_SCHEMA }
            )
            
            result = json.loads(response.choices[0].message.content)
            selected = result.get("selected_question")
            reason = result.get("reason", "No reason provided")
            
            return (selected is not None, reason, selected)
        except Exception as e:
            logger.error(f"Error evaluating questions: {str(e)}")
            return False, "Error in evaluation", None

def main():
    logger.info("Starting dataset processing")
    dataset = load_dataset("marianbasti/boletin-oficial-argentina", split="train")
    logger.info("Dataset loaded with %d entries", len(dataset))
    
    state = ProcessingState()
    
    stats = {
        "total_chunks": 0,
        "accepted_chunks": 0,
        "rejected_chunks": 0
    }
    
    dataset_length = len(dataset)
    
    # Initialize last_processed_idx if None
    if state.last_processed_idx is None:
        state.last_processed_idx = dataset_length
    
    mode = "a" if state.last_processed_idx < dataset_length else "w"
    try:
        with open("BORA-Qs.jsonl", mode, encoding="utf-8") as f:
            # Iterate through dataset in reverse order
            for idx in range(dataset_length - 1, -1, -1):
                logger.debug("Processing entry %d", idx)
                if idx >= state.last_processed_idx:
                    continue
                    
                entry = dataset[idx]
                chunks = chunk_text(entry["full_text"])
                stats["total_chunks"] += len(chunks)
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{idx}_{chunk_idx}"
                    logger.debug("Processing chunk %s", chunk_id)
                    if chunk_id in state.processed_chunks:
                        continue
                        
                    try:
                        # First evaluate just the content
                        is_valuable, reason = evaluate_content_quality(chunk)
                        
                        if not is_valuable:
                            stats["rejected_chunks"] += 1
                            logger.info(f"Skipping chunk {chunk_id}. Reason: {reason}")
                            continue
                        
                        # Generate two questions
                        questions = generate_question(chunk)
                        
                        # Evaluate and select the best question
                        is_question_valuable, question_reason, selected = evaluate_content_quality(chunk, questions)
                        
                        if not is_question_valuable:
                            stats["rejected_chunks"] += 1
                            logger.info(f"Rejecting questions for {chunk_id}. Reason: {question_reason}")
                            continue
                        
                        stats["accepted_chunks"] += 1
                        selected_question = questions[selected - 1]
                        
                        qa_pair = {
                            "id": chunk_id,
                            "context": chunk,
                            "question": selected_question["pregunta"]
                        }
                        
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                        f.flush()  # Ensure writing to disk
                        state.processed_chunks.add(chunk_id)
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                        continue
                
                state.last_processed_idx = idx
                state.save_state()
                
                if (dataset_length - idx) % 100 == 0:
                    logger.info(f"""Progress stats:
                        Processed entries: {dataset_length - idx}
                        Total chunks: {stats['total_chunks']}
                        Accepted chunks: {stats['accepted_chunks']}
                        Rejected chunks: {stats['rejected_chunks']}
                        Acceptance rate: {(stats['accepted_chunks']/stats['total_chunks']*100):.2f}%
                    """)
    except Exception as e:
        logger.critical("Fatal error in main processing loop: %s", str(e), exc_info=True)
        raise
    finally:
        logger.info("Processing completed. Final stats: %s", stats)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("Application failed: %s", str(e), exc_info=True)
        raise