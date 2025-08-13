import os
import pandas as pd
from dotenv import load_dotenv
import time
import logging
from datetime import datetime
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from regular_rag.pipeline import RegularRAGPipeline
from utils.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_comparison_execution.log'),
        logging.StreamHandler()
    ]
)

# Define Pydantic model for structured output (same as original)
class QuestionResponse(BaseModel):
    reasoning: str = Field(
        description="Your step-by-step reasoning process to arrive at the answer. Be thorough but concise."
    )
    final_answer: str = Field(
        description="The final answer to the question. Keep it concise (10 words or less). For multiple choice, just give the number. For ordering, give numbers in order (e.g., '1,3,2,4')."
    )

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Define retry decorator for API calls (same as original script)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_gemini_api(prompt, api_key):
    """Make API call to Gemini with retry logic and structured output (same as original)"""
    system_message = """You are a helpful assistant that provides structured responses to questions. 
You will provide:
1. Reasoning: Your step-by-step thought process to arrive at the answer
2. Final Answer: A concise answer (10 words or less) that matches the expected format

For multiple choice questions, just give the number of the correct answer.
For ordering questions, output the numbers in the correct order (e.g., "1,3,2,4").
Be direct and to the point in your final answer.

Respond with a JSON object containing 'reasoning' and 'final_answer' fields."""
    
    return completion(
        model="gemini/gemini-2.5-flash",  # Same model as original
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        api_key=api_key,
        response_format=QuestionResponse
    )

def prepare_documents_for_indexing(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Prepare documents from the dataset for RAG indexing.
    Each unique context becomes a document.
    """
    documents = []
    
    # Get unique contexts to avoid duplicate indexing
    unique_contexts = df[['context', 'doc_id']].drop_duplicates()
    
    for idx, row in unique_contexts.iterrows():
        doc = {
            "id": row['doc_id'],
            "content": row['context'],
            "kb_id": "loogle_dataset",
            "is_relevant": True,  # All documents are relevant for this experiment
            "question_id": -1,  # Not specific to any question since this is the full context
        }
        documents.append(doc)
    
    print(f"Prepared {len(documents)} unique documents from {len(df)} questions")
    return documents

def create_question_prompt(question: str, context: str) -> str:
    """
    Create the same prompt structure as the original script but with RAG-retrieved context.
    This mimics the original: {context}\n==========\nQuestion: {question}
    """
    return f"{context}\n==========\nQuestion: {question}"

def main():
    # Load the original dataset (same as original script)
    print("Loading dataset...")
    original_df = pd.read_csv('data/loogle_questions_with_short_answers.csv')
    
    # Limit to 2 examples for this comparison
    n_examples = 2
    df = original_df.head(n_examples).copy()
    print(f"Processing {n_examples} examples for RAG comparison...")
    
    # Prepare documents for indexing
    print("Preparing documents for RAG indexing...")
    documents = prepare_documents_for_indexing(original_df)  # Index all contexts, not just the 2 examples
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag_pipeline = RegularRAGPipeline(persist_directory="chroma_db_comparison")
    
    # Index documents
    print("Indexing documents...")
    indexing_stats = rag_pipeline.index_documents(documents)
    print(f"Indexing completed: {indexing_stats}")
    
    # Add new columns for tracking RAG results (similar to original script)
    df['execution_time_seconds'] = None
    df['actual_cost_usd'] = None
    df['prompt_tokens'] = None
    df['completion_tokens'] = None
    df['total_tokens'] = None
    df['rag_retrieval_time'] = None
    df['rag_chunks_retrieved'] = None
    df['rag_context_length'] = None
    df['rag_results'] = None
    df['reasoning'] = None
    df['final_answer'] = None
    df['success'] = None
    df['error'] = None
    
    # Track total costs and timing
    total_execution_time = 0
    total_cost = 0
    
    # Process each question using RAG for retrieval + LiteLLM for answer generation
    for i in range(len(df)):
        question = df.loc[i, 'question']
        print(f"\nProcessing question {i+1}/{len(df)}: {question[:100]}...")
        logging.info(f"Processing question {i+1}/{len(df)}: {question[:100]}...")
        
        try:
            # Start timing
            start_time = time.time()
            
            # Use RAG pipeline for retrieval only
            # We'll extract the context and use our own LLM call for fair comparison
            print("🔍 Retrieving relevant context using RAG...")
            retrieval_start = time.time()
            
            # Use the retriever directly to get context without generating an answer
            retrieval_result = rag_pipeline.retriever.retrieve(question, k=config.retrieval_k)
            retrieved_chunks = retrieval_result["chunks"]
            
            # Create context from retrieved chunks (same format as RAG pipeline)
            context_parts = []
            for j, chunk in enumerate(retrieved_chunks):
                context_parts.append(
                    f"--- Document {j + 1} (Similarity: {chunk['similarity']:.3f}) ---"
                )
                context_parts.append(chunk["content"])
                context_parts.append("")  # Empty line for separation
            
            rag_context = "\n".join(context_parts)
            retrieval_time = time.time() - retrieval_start
            
            print(f"📄 Retrieved {len(retrieved_chunks)} chunks ({len(rag_context)} chars) in {retrieval_time:.2f}s")
            
            # Create prompt using same format as original: context + separator + question
            prompt = f"{rag_context}\n==========\nQuestion: {question}"
            
            # Use LiteLLM for answer generation (same as original script)
            print("🧠 Generating answer using LiteLLM...")
            llm_start = time.time()
            response = call_gemini_api(prompt, api_key)
            llm_time = time.time() - llm_start
            
            # End timing
            end_time = time.time()
            execution_time = end_time - start_time
            total_execution_time += execution_time
            
            # Extract the structured response (same parsing as original script)
            if hasattr(response.choices[0].message, 'parsed') and response.choices[0].message.parsed:
                # Parse the structured output directly from LiteLLM
                structured_response = response.choices[0].message.parsed
                reasoning = structured_response.reasoning
                final_answer = structured_response.final_answer
                
                # Combine for backward compatibility
                answer = f"Reasoning: {reasoning}\nFinal Answer: {final_answer}"
            else:
                # Fallback to regular content if structured output fails
                content = response.choices[0].message.content
                try:
                    # Try to parse JSON from content
                    import json
                    parsed_content = json.loads(content)
                    reasoning = parsed_content.get('reasoning', 'No reasoning provided')
                    final_answer = parsed_content.get('final_answer', content)
                    answer = f"Reasoning: {reasoning}\nFinal Answer: {final_answer}"
                except (json.JSONDecodeError, TypeError):
                    # Use content as-is if JSON parsing fails
                    answer = content
                    reasoning = "Structured output failed, using fallback"
                    final_answer = content
            
            # Debug: Check if answer is empty or None (same as original script)
            if not final_answer or final_answer.strip() == "":
                logging.warning(f"Question {i+1} - Empty response received from API")
                answer = "EMPTY_RESPONSE"
                reasoning = "EMPTY_RESPONSE"
                final_answer = "EMPTY_RESPONSE"
            
            # Get actual cost and token info from LiteLLM response (same as original script)
            actual_cost = response._hidden_params.get('response_cost', 'N/A')
            prompt_tokens = response._hidden_params.get('prompt_tokens', 'N/A')
            completion_tokens = response._hidden_params.get('completion_tokens', 'N/A')
            total_tokens = response._hidden_params.get('total_tokens', 'N/A')
            
            # Format cost for display and add to total
            if isinstance(actual_cost, (int, float)):
                total_cost += actual_cost
                cost_str = f"${actual_cost:.6f}"
            else:
                cost_str = str(actual_cost)
            
            # Log details
            logging.info(f"Question {i+1} - Execution time: {execution_time:.2f} seconds")
            logging.info(f"Question {i+1} - Retrieval time: {retrieval_time:.2f} seconds")
            logging.info(f"Question {i+1} - LLM time: {llm_time:.2f} seconds")
            logging.info(f"Question {i+1} - Actual cost from LiteLLM: {cost_str}")
            logging.info(f"Question {i+1} - Tokens: {prompt_tokens} input, {completion_tokens} output, {total_tokens} total")
            logging.info(f"Question {i+1} - Final Answer: {final_answer}")
            
            # Store results
            df.at[i, 'rag_results'] = answer
            df.at[i, 'reasoning'] = reasoning
            df.at[i, 'final_answer'] = final_answer
            df.at[i, 'execution_time_seconds'] = execution_time
            df.at[i, 'actual_cost_usd'] = actual_cost if isinstance(actual_cost, (int, float)) else None
            df.at[i, 'prompt_tokens'] = prompt_tokens
            df.at[i, 'completion_tokens'] = completion_tokens
            df.at[i, 'total_tokens'] = total_tokens
            df.at[i, 'rag_retrieval_time'] = retrieval_time
            df.at[i, 'rag_chunks_retrieved'] = len(retrieved_chunks)
            df.at[i, 'rag_context_length'] = len(rag_context)
            df.at[i, 'success'] = True
            df.at[i, 'error'] = None
            
            print(f"✓ Question {i+1} completed in {execution_time:.2f}s")
            print(f"🔍 Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.2f}s")
            print(f"📏 Context length: {len(rag_context)} chars")
            print(f"💰 Actual cost: {cost_str}")
            print(f"🧠 Tokens: {prompt_tokens} input, {completion_tokens} output")
            print(f"⏱️  LLM time: {llm_time:.2f}s")
            print(f"✅ Final Answer: {final_answer}")
            
            # Save progress after each question
            output_file = 'data/loogle_rag_comparison_results.csv'
            df.to_csv(output_file, index=False)
            print(f"💾 Progress saved after question {i+1}")
            
            # Add a small delay to be respectful to APIs
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error processing question {i+1}: {e}")
            print(f"✗ Error processing question {i+1}: {e}")
            df.at[i, 'rag_results'] = f"ERROR: {str(e)}"
            df.at[i, 'reasoning'] = f"ERROR: {str(e)}"
            df.at[i, 'final_answer'] = f"ERROR: {str(e)}"
            df.at[i, 'execution_time_seconds'] = time.time() - start_time
            df.at[i, 'actual_cost_usd'] = None
            df.at[i, 'prompt_tokens'] = None
            df.at[i, 'completion_tokens'] = None
            df.at[i, 'total_tokens'] = None
            df.at[i, 'rag_retrieval_time'] = None
            df.at[i, 'rag_chunks_retrieved'] = None
            df.at[i, 'rag_context_length'] = None
            df.at[i, 'success'] = False
            df.at[i, 'error'] = str(e)
            
            # Save progress after each error
            output_file = 'data/loogle_rag_comparison_results.csv'
            df.to_csv(output_file, index=False)
            print(f"💾 Progress saved after error on question {i+1}")
    
    # Final save
    output_file = 'data/loogle_rag_comparison_results.csv'
    df.to_csv(output_file, index=False)
    
    # Log summary (same as original script)
    logging.info(f"RAG comparison completed. Total execution time: {total_execution_time:.2f} seconds")
    if isinstance(total_cost, (int, float)):
        logging.info(f"Total actual cost from LiteLLM: ${total_cost:.6f}")
    else:
        logging.info(f"Total actual cost from LiteLLM: {total_cost}")
    logging.info("Note: LiteLLM provides actual billing costs from the API response")
    
    print(f"\n✅ RAG comparison completed!")
    print(f"📊 Results saved to '{output_file}'")
    print(f"⏱️  Total execution time: {total_execution_time:.2f} seconds")
    if isinstance(total_cost, (int, float)):
        print(f"💰 Total actual cost from LiteLLM: ${total_cost:.6f}")
    else:
        print(f"💰 Total actual cost from LiteLLM: {total_cost}")
    print(f"🏗️  Indexing stats: {indexing_stats}")
    print(f"📝 Detailed logs saved to 'rag_comparison_execution.log'")
    print(f"🔧 Using RAG with {config.retrieval_k} retrieved chunks per question")
    print(f"ℹ️  Note: LiteLLM provides actual billing costs from the API response")
    print(f"🔄 Using same model (gemini/gemini-2.5-flash) and structured output as original script")

if __name__ == "__main__":
    main()
