import os
from datasets import load_dataset
from openai import OpenAI
import json
import logging
from logging.handlers import RotatingFileHandler
import time
from functools import wraps
import pickle

# Enhanced logging configuration
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
    def __init__(self, state_file="processing_state.pkl"):
        self.state_file = state_file
        self.last_processed_idx = None  # Changed to None as default
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
QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "pregunta": {
            "type": "string",
            "description": "Una pregunta relevante basada en el contexto proporcionado"
        }
    },
    "required": ["pregunta"]
}

@exponential_backoff_retry(max_retries=None)
def generate_question(context):
    """Generate a question based on the context."""
    logger.debug("Generating question for context of length %d", len(context))
    prompt = f"""Actúa como un abogado buscando información legal.
            Genera una pregunta natural que:
            1. Sea autocontenida y no haga referencia al texto o contexto
            2. Use lenguaje cotidiano mezclado con términos legales relevantes
            3. Represente una duda real sobre el tema principal del texto
            4. Sea clara y específica, evitando ambigüedades o generalidades
            5. Evite frases como "según el texto", "de acuerdo a la ley", "en este caso", etc.
            
            Ejemplos de buenas preguntas:
            ✓ "¿Cuáles son los requisitos para registrar una marca comercial?"
            ✓ "¿Qué documentación necesito para iniciar un trámite de jubilación?"
            ✓ "¿Cómo funciona el proceso de declaración de herederos?"
            
            Ejemplos de malas preguntas:
            ✗ "Según esta normativa, ¿qué documentos se requieren?"
            ✗ "De acuerdo al texto, ¿cuál es el procedimiento?"
            ✗ "¿Qué establece esta ley sobre el tema?"
            
            Fragmento a analizar: {context}
            
            Responde con un JSON que contenga la pregunta"""

    try:
        response = client.chat.completions.create(
            model="",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={ "type": "json_object", "json_schema": QUESTION_SCHEMA }
        )
        
        content = response.choices[0].message.content
        logger.info(f"LLM Response: {content}")
        
        result = json.loads(content)
        if "pregunta" not in result:
            logger.error(f"Invalid response format. Expected 'pregunta' key. Got: {result}")
            return "¿Qué información relevante contiene este texto?"  # fallback question
            
        return result["pregunta"]
    except Exception as e:
        logger.error(f"Error generating question: {str(e)}")
        return "¿Qué información relevante contiene este texto?"  # fallback question

@exponential_backoff_retry(max_retries=None)
def evaluate_content_quality(context, question=None):
    """Evaluate if the content and question (if provided) are suitable for the dataset."""
    
    # Context-only evaluation prompt
    context_prompt = """Actúa como un evaluador experto en calidad de datos para sistemas RAG.
    Analiza si este fragmento de texto legal es adecuado como fuente de información según estos criterios:

    Criterios de evaluación del contenido:
    1. Sustancia legal: contiene normas, procedimientos o conceptos legales relevantes
    2. Aplicabilidad práctica: aborda situaciones o consultas realistas
    3. Claridad: la información está expresada de manera estructurada
    4. Potencial de consulta: podría responder a consultas específicas
    
    Contexto a evaluar: {context}
    
    Responde con un JSON que contenga:
    {{
        "is_valuable": true/false,
        "reason": "Explicación de la decisión"
    }}"""

    # Question evaluation prompt
    question_prompt = """Actúa como un evaluador experto en calidad de datos para sistemas RAG.
    Analiza si esta pregunta es adecuada para entrenar un sistema RAG según estos criterios:

    Criterios de evaluación de la pregunta:
    1. Naturalidad: refleja consultas reales de abogados
    2. Utilidad pedagógica: ayuda al modelo a aprender asociaciones semánticas
    
    IMPORTANTE: Rechazar preguntas que contengan frases como "según el texto", 
    "de acuerdo a la ley", "en este caso", o similares referencias al contexto.
    
    Contexto: {context}
    Pregunta: {question}
    
    Responde con un JSON que contenga:
    {{
        "is_valuable": true/false,
        "reason": "Explicación detallada de la decisión"
    }}"""

    try:
        if question is None:
            # Evaluate just the context
            response = client.chat.completions.create(
                model="",
                messages=[{
                    "role": "user", 
                    "content": context_prompt.format(context=context)
                }],
                temperature=0.3,
                response_format={ "type": "json_object" }
            )
        else:
            # Evaluate the question
            response = client.chat.completions.create(
                model="",
                messages=[{
                    "role": "user", 
                    "content": question_prompt.format(
                        context=context,
                        question=question
                    )
                }],
                temperature=0.3,
                response_format={ "type": "json_object" }
            )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("is_valuable", False), result.get("reason", "No reason provided")
    except Exception as e:
        logger.error(f"Error evaluating quality: {str(e)}")
        return False, "Error in evaluation"

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
    
    # Get total length of dataset
    dataset_length = len(dataset)
    
    # Initialize last_processed_idx if None
    if state.last_processed_idx is None:
        state.last_processed_idx = dataset_length
    
    mode = "a" if state.last_processed_idx < dataset_length else "w"
    try:
        with open("spanish_qa_dataset.jsonl", mode, encoding="utf-8") as f:
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
                        
                        # Generate question
                        question = generate_question(chunk)
                        
                        # Evaluate the question
                        is_question_valuable, question_reason = evaluate_content_quality(chunk, question)
                        
                        if not is_question_valuable:
                            stats["rejected_chunks"] += 1
                            logger.info(f"Rejecting question {chunk_id}. Reason: {question_reason}")
                            continue
                        
                        stats["accepted_chunks"] += 1
                        qa_pair = {
                            "id": chunk_id,
                            "context": chunk,
                            "question": question
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