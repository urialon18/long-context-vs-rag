import os
import pandas as pd
from dotenv import load_dotenv
import time
import logging
import argparse
from datetime import datetime
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemini_api_execution.log'),
        logging.StreamHandler()
    ]
)

# Custom exception for empty responses
class EmptyResponseError(Exception):
    """Raised when the LLM returns an empty response"""
    pass

# No structured output model needed - using regular text responses

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run context-in-prompt evaluation with configurable model')
parser.add_argument('--model', type=str, default='gemini/gemini-2.5-flash', 
                    help='Model to use for evaluation (default: gemini/gemini-2.5-flash)')
parser.add_argument('--k', type=int, default=20,
                    help='Number of questions to process (default: 20)')
parser.add_argument('--max-retries', type=int, default=3,
                    help='Maximum number of retries for API calls (default: 3)')
parser.add_argument('--reasoning-effort', type=str, default='medium',
                    choices=['disable', 'low', 'medium', 'high'],
                    help='Reasoning effort level for Gemini models (default: medium). Maps to budget tokens: disable=0, low=1024, medium=2048, high=4096')
parser.add_argument('--enable-caching', action='store_true',
                    help='Enable context caching for Gemini models (requires paid tier)')
args = parser.parse_args()

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configuration
MODEL = args.model
MAX_RETRIES = args.max_retries
K = args.k
REASONING_EFFORT = args.reasoning_effort
ENABLE_CACHING = args.enable_caching

print(f"Using model: {MODEL}")
print(f"Max retries: {MAX_RETRIES}")
print(f"Number of questions to process: {K}")
print(f"Reasoning effort: {REASONING_EFFORT}")
print(f"Context caching: {'Enabled' if ENABLE_CACHING else 'Disabled'}")
logging.info(f"Configuration - Model: {MODEL}, Max retries: {MAX_RETRIES}, K: {K}, Reasoning effort: {REASONING_EFFORT}, Caching: {ENABLE_CACHING}")

# Load the original dataset
original_df = pd.read_csv('data/loogle_questions_with_short_answers.csv')

# Check if output file already exists and load it to preserve existing results
# Replace slashes in model name to create valid filename and include reasoning effort
model_filename = MODEL.replace('/', '_').replace('\\', '_')
output_file = f'data/loogle_short_answers_full_context_llm_{model_filename}_{REASONING_EFFORT}.csv'
if os.path.exists(output_file):
    print(f"Loading existing results from {output_file}")
    existing_df = pd.read_csv(output_file)
    
    # Start with original dataset and merge existing results
    df = original_df.copy()
    
    # Add new columns for tracking metrics
    for col in ['execution_time_seconds', 'actual_cost_usd', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'full_context_results', 'reasoning', 'final_answer']:
        df[col] = None
    
    # Merge existing results based on question ID
    if 'id' in existing_df.columns and len(existing_df) > 0:
        # Match by id and update existing results
        for idx, row in df.iterrows():
            question_id = row['id']
            existing_row = existing_df[existing_df['id'] == question_id]
            if len(existing_row) > 0:
                existing_row = existing_row.iloc[0]
                for col in ['execution_time_seconds', 'actual_cost_usd', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'full_context_results', 'reasoning', 'final_answer']:
                    if col in existing_row and pd.notna(existing_row[col]):
                        df.at[idx, col] = existing_row[col]
else:
    # Start with original dataset and add new columns
    df = original_df.copy()
    # Add new columns for tracking metrics
    df['execution_time_seconds'] = None
    df['actual_cost_usd'] = None
    df['prompt_tokens'] = None
    df['completion_tokens'] = None
    df['total_tokens'] = None
    df['full_context_results'] = None
    df['reasoning'] = None
    df['final_answer'] = None

# Process all questions in the dataset
k = min(K, len(df))
print(f"Processing all {k} questions in the dataset...")
logging.info(f"Starting processing of {k} questions using LiteLLM with structured output")

# Track total costs
total_cost = 0
total_execution_time = 0

# Count how many questions are already completed
completed_count = df.head(k)['execution_time_seconds'].notna().sum()
print(f"Found {completed_count} already completed questions out of {k}")
logging.info(f"Found {completed_count} already completed questions out of {k}")

# Define retry decorator for API calls (using configurable max retries)
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception, EmptyResponseError)),
    reraise=True
)
def call_gemini_api(prompt, api_key, model, reasoning_effort, enable_caching=False):
    """Make API call to Gemini with retry logic, regular text output, and optional context caching"""
    system_message = """You are a helpful assistant that provides concise answers to questions.

The final answer to the question should be concise (10 words or less). For multiple choice questions, just give the number of the correct answer. For ordering questions, give numbers in order (e.g., '1,3,2,4').

Be direct and to the point. Only provide the answer, no additional explanation or reasoning."""
    
    # Prepare user message with optional caching for Gemini models
    user_message = {"role": "user", "content": prompt}
    
    # Add context caching for Gemini models if enabled (requires paid tier)
    if model.startswith('gemini/') and enable_caching:
        user_message["content"] = [
            {
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral", "ttl": "3600s"}  # Cache for 1 hour
            }
        ]
    
    # Prepare completion parameters
    completion_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            user_message
        ],
        "api_key": api_key
    }
    
    # Add reasoning_effort parameter for Gemini models
    if model.startswith('gemini/'):
        completion_params["reasoning_effort"] = reasoning_effort
    
    response = completion(**completion_params)
    
    # Check for empty response and raise exception if found
    content = response.choices[0].message.content
    if not content or content.strip() == "":
        raise EmptyResponseError("Received empty content from API response")
    
    return response

for i in range(k):
    question = df.loc[i, 'question']
    
    # Check if this question is already completed using execution_time_seconds as the main indicator
    if pd.notna(df.loc[i, 'execution_time_seconds']):
        print(f"\n⏭️  Skipping question {i+1}/{k} (already completed): {question[:100]}...")
        logging.info(f"Skipping question {i+1}/{k} (already completed): {question[:100]}...")
        continue
    
    print(f"\nProcessing question {i+1}/{k}: {question[:100]}...")
    logging.info(f"Processing question {i+1}/{k}: {question[:100]}...")
    
    # Get the context for this specific question
    question_context = df.loc[i, 'context']
    
    # Combine context and question
    prompt = f"{question_context}\n==========\nQuestion: {question}"
    
    try:
        # Start timing
        start_time = time.time()
        
        # Generate content using LiteLLM with configurable model with retry logic and structured output
        response = call_gemini_api(prompt, api_key, MODEL, REASONING_EFFORT, ENABLE_CACHING)
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
        
        # Extract the regular text response
        content = response.choices[0].message.content
        final_answer = content.strip()
        
        # Use the content as both the answer and final_answer for backward compatibility
        answer = final_answer
        
        # Note: Empty response checking is now handled in the retry mechanism
        
        # Get actual cost from LiteLLM response
        actual_cost = response._hidden_params.get('response_cost', 'N/A')
        
        # Get token information if available
        prompt_tokens = response._hidden_params.get('prompt_tokens', 'N/A')
        completion_tokens = response._hidden_params.get('completion_tokens', 'N/A')
        total_tokens = response._hidden_params.get('total_tokens', 'N/A')
        
        # Add cost to total if it's a number
        if isinstance(actual_cost, (int, float)):
            total_cost += actual_cost
            cost_str = f"${actual_cost:.6f}"
        else:
            cost_str = str(actual_cost)
        
        logging.info(f"Question {i+1} - Execution time: {execution_time:.2f} seconds")
        logging.info(f"Question {i+1} - Actual cost from LiteLLM: {cost_str}")
        logging.info(f"Question {i+1} - Tokens: {prompt_tokens} input, {completion_tokens} output, {total_tokens} total")
        logging.info(f"Question {i+1} - Final Answer: {final_answer}")
        
        # Note about internal caching
        if i > 0:
            logging.info(f"Question {i+1} - Note: Gemini has internal caching that may provide token discounts for repeated content")
        
        # Add the results to the DataFrame
        df.at[i, 'full_context_results'] = answer
        df.at[i, 'reasoning'] = "No reasoning (using direct answer mode)"
        df.at[i, 'final_answer'] = final_answer
        df.at[i, 'execution_time_seconds'] = execution_time
        df.at[i, 'actual_cost_usd'] = actual_cost if isinstance(actual_cost, (int, float)) else None
        df.at[i, 'prompt_tokens'] = prompt_tokens
        df.at[i, 'completion_tokens'] = completion_tokens
        df.at[i, 'total_tokens'] = total_tokens
        
        print(f"✓ Question {i+1} completed in {execution_time:.2f}s")
        print(f"💰 Actual cost: {cost_str}")
        print(f"✅ Final Answer: {final_answer}")
        
        # Save progress after each successful completion
        df.to_csv(output_file, index=False)
        print(f"💾 Progress saved after question {i+1}")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        
    except EmptyResponseError as e:
        logging.error(f"Failed to get valid response for question {i+1} after {MAX_RETRIES} retries: {e}")
        print(f"✗ Failed to get valid response for question {i+1} after {MAX_RETRIES} retries: {e}")
        df.at[i, 'full_context_results'] = f"EMPTY_RESPONSE_ERROR: {str(e)}"
        df.at[i, 'reasoning'] = "No reasoning (error occurred)"
        df.at[i, 'final_answer'] = f"EMPTY_RESPONSE_ERROR: {str(e)}"
        # Set error values for metrics
        df.at[i, 'execution_time_seconds'] = None
        df.at[i, 'actual_cost_usd'] = None
        df.at[i, 'prompt_tokens'] = None
        df.at[i, 'completion_tokens'] = None
        df.at[i, 'total_tokens'] = None
        
        # Save progress after each error to preserve what we have
        df.to_csv(output_file, index=False)
        print(f"💾 Progress saved after empty response error on question {i+1}")
        
    except Exception as e:
        logging.error(f"Error processing question {i+1} after {MAX_RETRIES} retries: {e}")
        print(f"✗ Error processing question {i+1} after {MAX_RETRIES} retries: {e}")
        df.at[i, 'full_context_results'] = f"ERROR: {str(e)}"
        df.at[i, 'reasoning'] = "No reasoning (error occurred)"
        df.at[i, 'final_answer'] = f"ERROR: {str(e)}"
        # Set error values for metrics
        df.at[i, 'execution_time_seconds'] = None
        df.at[i, 'actual_cost_usd'] = None
        df.at[i, 'prompt_tokens'] = None
        df.at[i, 'completion_tokens'] = None
        df.at[i, 'total_tokens'] = None
        
        # Save progress after each error to preserve what we have
        df.to_csv(output_file, index=False)
        print(f"💾 Progress saved after error on question {i+1}")

# Log summary
logging.info(f"Processing completed. Total execution time: {total_execution_time:.2f} seconds")
if isinstance(total_cost, (int, float)):
    logging.info(f"Total actual cost from LiteLLM: ${total_cost:.6f}")
else:
    logging.info(f"Total actual cost from LiteLLM: {total_cost}")
logging.info("Note: LiteLLM provides actual billing costs from the API response")

# Save the updated DataFrame to a CSV file
df.to_csv(output_file, index=False)

print(f"\n✅ Results saved to '{output_file}'")
print(f"✅ Direct responses have been added:")
print(f"   - 'full_context_results' column: Direct answer from model")
print(f"   - 'reasoning' column: Placeholder (direct answer mode)")
print(f"   - 'final_answer' column: Concise final answer only")
print(f"📊 Total execution time: {total_execution_time:.2f} seconds")
if isinstance(total_cost, (int, float)):
    print(f"💰 Total actual cost from LiteLLM: ${total_cost:.6f}")
else:
    print(f"💰 Total actual cost from LiteLLM: {total_cost}")
print(f"📝 Detailed logs saved to 'gemini_api_execution.log'")
print(f"ℹ️  Note: LiteLLM provides actual billing costs from the API response")
print(f"🔧 Using direct text output with concise system prompt guidance")
print(f"🤖 Model used: {MODEL}")
print(f"🔄 Max retries: {MAX_RETRIES} (includes retry for empty responses)")
print(f"🧠 Reasoning effort: {REASONING_EFFORT} (for Gemini models)")
if ENABLE_CACHING:
    print(f"⚡ Context caching: Enabled for Gemini models (1 hour TTL)")
else:
    print(f"⚡ Context caching: Disabled (use --enable-caching for paid tier)")
