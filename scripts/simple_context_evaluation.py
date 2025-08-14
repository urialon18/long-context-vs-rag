#!/usr/bin/env python3
"""
Simple script to evaluate questions using full context in LLM prompts.
Now supports both long context and regular RAG evaluation for comparison.
Clean, function-based approach with minimal overhead.
"""

import argparse
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from regular_rag.pipeline import RegularRAGPipeline


class AnswerComparison(BaseModel):
    """Structured output for answer comparison"""

    is_correct: bool


def load_environment():
    """Load environment variables and validate API key."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    return api_key


def prepare_documents_for_rag(df):
    """Prepare documents from the dataset for RAG indexing."""
    documents = []

    # Only index unique contexts to prevent duplicates
    unique_docs = df.drop_duplicates(subset=["doc_id"]).copy()

    print(
        f"📚 Indexing {len(unique_docs)} unique documents (from {len(df)} total questions)"
    )

    for _, row in unique_docs.iterrows():
        documents.append(
            {
                "id": row["doc_id"],  # Use doc_id as the unique identifier
                "content": row["context"],
                "metadata": {
                    "doc_id": row["doc_id"],
                    "task": row["task"],
                    "title": row.get("title", ""),
                },
            }
        )
    return documents


def load_questions_data(
    filepath="data/loogle_questions_with_short_answers.csv", output_file=None
):
    """Load the questions dataset and prepare for processing."""
    # Load original dataset
    original_df = pd.read_csv(filepath)

    # Check if output file already exists and load existing results
    if output_file and os.path.exists(output_file):
        print(f"Found existing results file: {output_file}")
        existing_df = pd.read_csv(output_file)

        # Start with original dataset
        df = original_df.copy()

        # Add new columns if they don't exist
        for col in [
            "llm_answer",
            "actual_cost_usd",
            "response_time_seconds",
            "is_correct",
            "retrieved_context",
        ]:
            if col not in df.columns:
                df[col] = None

        # Merge existing results based on index/id
        if len(existing_df) > 0 and "llm_answer" in existing_df.columns:
            # Match by index position and update existing results
            for idx in existing_df.index:
                if idx < len(df):
                    for col in [
                        "llm_answer",
                        "actual_cost_usd",
                        "response_time_seconds",
                        "is_correct",
                        "retrieved_context",
                    ]:
                        if col in existing_df.columns and pd.notna(
                            existing_df.loc[idx, col]
                        ):
                            df.at[idx, col] = existing_df.loc[idx, col]

        # Count existing completed questions
        completed_count = df["llm_answer"].notna().sum()
        print(f"Found {completed_count} already completed questions")

    else:
        # Start fresh with original dataset
        df = original_df.copy()
        # Add only the essential columns we need
        df["llm_answer"] = None
        df["actual_cost_usd"] = None
        df["response_time_seconds"] = None
        df["is_correct"] = None
        df["retrieved_context"] = None
        print("Starting fresh - no existing results found")

    return df


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def call_llm_with_retry(
    prompt, api_key, model="gemini/gemini-2.5-flash", reasoning_effort="high"
):
    """Make API call to LLM with retry logic and reasoning effort."""
    system_message = """You are a helpful assistant that provides concise answers to questions. The final answer to the question should be concise (10 words or less). For multiple choice questions, just give the number of the correct answer. For ordering questions, give numbers in order (e.g., '1,3,2,4')."""

    completion_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "api_key": api_key,
    }

    # Add reasoning_effort parameter if supported by the model and not disabled
    if reasoning_effort and reasoning_effort != "disable":
        completion_params["reasoning_effort"] = reasoning_effort

    response = completion(**completion_params)

    # Validate response
    content = response.choices[0].message.content
    if not content or content.strip() == "":
        raise ValueError("Received empty content from API response")

    return response


def create_prompt(context, question):
    """Create the prompt by combining context and question."""
    return f"{context}\n==========\nQuestion: {question}"


def extract_response_data(response):
    """Extract the answer and cost from the LLM response."""
    final_answer = response.choices[0].message.content.strip()
    actual_cost = response._hidden_params.get("response_cost", None)
    return final_answer, actual_cost


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def compare_answers_with_llm(
    ground_truth, llm_answer, api_key, model="gemini/gemini-2.5-flash"
):
    """Compare ground truth answer with LLM answer using structured output."""
    comparison_prompt = f"""Compare these two answers to determine if they are equivalent:

Ground Truth Answer: {ground_truth}
LLM Answer: {llm_answer}

Determine if the LLM answer is correct compared to the ground truth. Consider:
- Exact matches
- Semantically equivalent answers
- For multiple choice: same option number
- For numerical: same value (accounting for reasonable precision)

Return True if the answers are equivalent/correct, False otherwise."""

    response = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert evaluator comparing answers for correctness.",
            },
            {"role": "user", "content": comparison_prompt},
        ],
        api_key=api_key,
        response_format=AnswerComparison,
    )

    # Handle different response formats
    if (
        hasattr(response.choices[0].message, "parsed")
        and response.choices[0].message.parsed
    ):
        return response.choices[0].message.parsed.is_correct
    else:
        # Fallback: parse JSON content manually
        import json

        content = response.choices[0].message.content
        try:
            result = json.loads(content)
            return result.get("is_correct", False)
        except (json.JSONDecodeError, AttributeError):
            # Final fallback: simple text parsing
            content_lower = str(content).lower()
            return "true" in content_lower and "false" not in content_lower


def process_questions_regular_rag(
    df, api_key, model, max_questions=None, reasoning_effort="high", output_file=None
):
    """Process questions using Regular RAG and get LLM responses."""
    total_questions = len(df) if max_questions is None else min(max_questions, len(df))
    total_cost = 0

    # Count already completed questions
    already_completed = df.head(total_questions)["llm_answer"].notna()
    completed_count = already_completed.sum()

    print(
        f"Processing {total_questions} questions with Regular RAG, reasoning_effort='{reasoning_effort}'..."
    )
    if completed_count > 0:
        print(
            f"Resuming from question {completed_count + 1} ({completed_count} already completed)"
        )

    # Initialize Regular RAG pipeline
    print("🔄 Initializing Regular RAG pipeline...")
    rag_pipeline = RegularRAGPipeline()

    # Prepare and index documents if not already indexed
    if not rag_pipeline.is_indexed:
        print("📚 Indexing documents for Regular RAG...")
        documents = prepare_documents_for_rag(df)
        rag_pipeline.index_documents(documents)

    # Create progress bar
    with tqdm(total=total_questions, desc="Processing RAG", unit="question") as pbar:
        for i in range(total_questions):
            # Check if this question is already completed
            if pd.notna(df.loc[i, "llm_answer"]):
                # Update progress for already completed questions
                current_correct = df.head(i + 1)["is_correct"].sum()
                current_success_rate = (current_correct / (i + 1)) * 100
                pbar.set_postfix(
                    {
                        "Success": f"{current_success_rate:.1f}%",
                        "Correct": f"{int(current_correct)}/{i + 1}",
                    }
                )
                pbar.update(1)
                continue

            question = df.loc[i, "question"]
            answer = df.loc[i, "answer"]
            ground_truth = f"Question: {question}\nAnswer: {answer}"

            try:
                # Use Regular RAG to answer the question
                start_time = time.time()
                rag_result = rag_pipeline.answer_question(question, question_id=i)
                end_time = time.time()
                response_time = end_time - start_time

                # Extract the retrieved context that was actually used
                retrieved_context = rag_result.get("context", "")

                # Extract the answer from RAG result
                # For multiple choice questions, try to get the predicted answer (number), otherwise use the full response
                if rag_result["predicted_answer"] is not None:
                    llm_answer = str(rag_result["predicted_answer"])
                else:
                    # For non-multiple choice questions, use the raw response but try to extract a concise answer
                    llm_answer = rag_result["llm_response"].strip()
                    # If it's too long, try to extract the last sentence or a concise part
                    if len(llm_answer) > 50:
                        sentences = llm_answer.split(".")
                        if len(sentences) > 1:
                            # Take the last non-empty sentence as it's likely the final answer
                            last_sentence = (
                                sentences[-2].strip()
                                if sentences[-1].strip() == ""
                                else sentences[-1].strip()
                            )
                            if len(last_sentence) < 50:
                                llm_answer = last_sentence

                # Compare answers using LLM
                is_correct = compare_answers_with_llm(
                    ground_truth, str(llm_answer), api_key, model
                )

                # Estimate cost (Regular RAG doesn't use litellm directly, so we estimate)
                # This is a rough estimation - in practice you'd want more accurate cost tracking
                estimated_cost = 0.001  # Placeholder cost estimation

                # Update dataframe
                df.at[i, "llm_answer"] = str(llm_answer)
                df.at[i, "actual_cost_usd"] = estimated_cost
                df.at[i, "response_time_seconds"] = response_time
                df.at[i, "is_correct"] = is_correct
                df.at[i, "retrieved_context"] = retrieved_context

                # Track total cost
                total_cost += estimated_cost

                # Calculate current cumulative success rate
                current_correct = df.head(i + 1)["is_correct"].sum()
                success_rate = (current_correct / (i + 1)) * 100
                pbar.set_postfix(
                    {
                        "Success": f"{success_rate:.1f}%",
                        "Correct": f"{int(current_correct)}/{i + 1}",
                    }
                )
                pbar.update(1)

                # Save progress after each successful completion
                if output_file:
                    df.to_csv(output_file, index=False)

                # Small delay to avoid rate limiting
                time.sleep(2)

            except Exception as e:
                df.at[i, "llm_answer"] = f"ERROR: {str(e)}"
                df.at[i, "actual_cost_usd"] = None
                df.at[i, "response_time_seconds"] = None
                df.at[i, "is_correct"] = False
                df.at[i, "retrieved_context"] = None

                # Update progress bar
                current_correct = df.head(i + 1)["is_correct"].sum()
                success_rate = (current_correct / (i + 1)) * 100
                pbar.set_postfix(
                    {
                        "Success": f"{success_rate:.1f}%",
                        "Correct": f"{int(current_correct)}/{i + 1}",
                    }
                )
                pbar.update(1)

                # Save progress after each error
                if output_file:
                    df.to_csv(output_file, index=False)

    # Calculate final statistics
    final_correct_count = df.head(total_questions)["is_correct"].sum()
    final_success_rate = (
        (final_correct_count / total_questions) * 100 if total_questions > 0 else 0
    )
    return total_cost, final_correct_count, total_questions, final_success_rate


def process_questions(
    df, api_key, model, max_questions=None, reasoning_effort="high", output_file=None
):
    """Process questions and get LLM responses using long context."""
    total_questions = len(df) if max_questions is None else min(max_questions, len(df))
    total_cost = 0

    # Count already completed questions
    already_completed = df.head(total_questions)["llm_answer"].notna()
    completed_count = already_completed.sum()

    print(
        f"Processing {total_questions} questions with Long Context, reasoning_effort='{reasoning_effort}'..."
    )
    if completed_count > 0:
        print(
            f"Resuming from question {completed_count + 1} ({completed_count} already completed)"
        )

    # Create progress bar
    with tqdm(
        total=total_questions, desc="Processing Long Context", unit="question"
    ) as pbar:
        for i in range(total_questions):
            # Check if this question is already completed
            if pd.notna(df.loc[i, "llm_answer"]):
                # Update progress for already completed questions
                current_correct = df.head(i + 1)["is_correct"].sum()
                current_success_rate = (current_correct / (i + 1)) * 100
                pbar.set_postfix(
                    {
                        "Success": f"{current_success_rate:.1f}%",
                        "Correct": f"{int(current_correct)}/{i + 1}",
                    }
                )
                pbar.update(1)
                continue

            question = df.loc[i, "question"]
            answer = df.loc[i, "answer"]
            ground_truth = f"Question: {question}\nAnswer: {answer}"

            # For long context, use ALL available contexts (like Regular RAG has access to all documents)
            all_contexts = df["context"].unique()
            combined_context = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join(all_contexts)

            try:
                # Create prompt and call LLM with timing
                prompt = create_prompt(combined_context, question)
                start_time = time.time()
                response = call_llm_with_retry(prompt, api_key, model, reasoning_effort)
                end_time = time.time()
                response_time = end_time - start_time

                # For long context, the retrieved context is all available contexts combined
                retrieved_context = combined_context

                # Extract data from response
                llm_answer, actual_cost = extract_response_data(response)

                # Compare answers using LLM
                is_correct = compare_answers_with_llm(
                    ground_truth, llm_answer, api_key, model
                )

                # Update dataframe
                df.at[i, "llm_answer"] = llm_answer
                df.at[i, "actual_cost_usd"] = actual_cost
                df.at[i, "response_time_seconds"] = response_time
                df.at[i, "is_correct"] = is_correct
                df.at[i, "retrieved_context"] = retrieved_context

                # Track success is now calculated dynamically from the dataframe

                # Track total cost
                if isinstance(actual_cost, (int, float)):
                    total_cost += actual_cost

                # Calculate current cumulative success rate
                current_correct = df.head(i + 1)["is_correct"].sum()
                success_rate = (current_correct / (i + 1)) * 100
                pbar.set_postfix(
                    {
                        "Success": f"{success_rate:.1f}%",
                        "Correct": f"{int(current_correct)}/{i + 1}",
                    }
                )
                pbar.update(1)

                # Save progress after each successful completion
                if output_file:
                    df.to_csv(output_file, index=False)

                # Small delay to avoid rate limiting
                time.sleep(1)

            except Exception as e:
                df.at[i, "llm_answer"] = f"ERROR: {str(e)}"
                df.at[i, "actual_cost_usd"] = None
                df.at[i, "response_time_seconds"] = None
                df.at[i, "is_correct"] = False
                df.at[i, "retrieved_context"] = None

                # Update progress bar
                current_correct = df.head(i + 1)["is_correct"].sum()
                success_rate = (current_correct / (i + 1)) * 100
                pbar.set_postfix(
                    {
                        "Success": f"{success_rate:.1f}%",
                        "Correct": f"{int(current_correct)}/{i + 1}",
                    }
                )
                pbar.update(1)

                # Save progress after each error
                if output_file:
                    df.to_csv(output_file, index=False)

    # Calculate final statistics
    final_correct_count = df.head(total_questions)["is_correct"].sum()
    final_success_rate = (
        (final_correct_count / total_questions) * 100 if total_questions > 0 else 0
    )
    return total_cost, final_correct_count, total_questions, final_success_rate


def main():
    """Main function to orchestrate the evaluation process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compare Long Context vs Regular RAG evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-2.5-flash",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=20,
        help="Maximum number of questions to process",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        choices=["low", "medium", "high", "disable"],
        help="Reasoning effort level for supported models (default: high)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["long-context", "regular-rag", "both"],
        help="Evaluation method: long-context, regular-rag, or both (default: both)",
    )
    args = parser.parse_args()

    print(f"Using model: {args.model}")
    print(f"Max questions: {args.max_questions}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Evaluation method: {args.method}")

    try:
        # Setup
        api_key = load_environment()
        model_filename = args.model.replace("/", "_").replace("\\", "_")
        os.makedirs("data/results", exist_ok=True)

        results = {}

        # Run Long Context evaluation
        if args.method in ["long-context", "both"]:
            print("\n🔥 Running Long Context Evaluation")
            print("=" * 50)

            # Calculate output filename
            long_context_output_file = f"data/results/long_context_results_{model_filename}_{args.reasoning_effort}.csv"

            # Load data (checking for existing results)
            df_long = load_questions_data(output_file=long_context_output_file)

            # Process questions with long context
            (
                total_cost_long,
                correct_count_long,
                total_questions_long,
                success_rate_long,
            ) = process_questions(
                df_long,
                api_key,
                args.model,
                args.max_questions,
                args.reasoning_effort,
                long_context_output_file,
            )

            # Final save
            df_long.to_csv(long_context_output_file, index=False)

            results["long_context"] = {
                "output_file": long_context_output_file,
                "correct_count": correct_count_long,
                "total_questions": total_questions_long,
                "success_rate": success_rate_long,
                "total_cost": total_cost_long,
            }

        # Run Regular RAG evaluation
        if args.method in ["regular-rag", "both"]:
            print("\n🔍 Running Regular RAG Evaluation")
            print("=" * 50)

            # Calculate output filename
            regular_rag_output_file = f"data/results/regular_rag_results_{model_filename}_{args.reasoning_effort}.csv"

            # Load data (checking for existing results)
            df_rag = load_questions_data(output_file=regular_rag_output_file)

            # Process questions with regular RAG
            total_cost_rag, correct_count_rag, total_questions_rag, success_rate_rag = (
                process_questions_regular_rag(
                    df_rag,
                    api_key,
                    args.model,
                    args.max_questions,
                    args.reasoning_effort,
                    regular_rag_output_file,
                )
            )

            # Final save
            df_rag.to_csv(regular_rag_output_file, index=False)

            results["regular_rag"] = {
                "output_file": regular_rag_output_file,
                "correct_count": correct_count_rag,
                "total_questions": total_questions_rag,
                "success_rate": success_rate_rag,
                "total_cost": total_cost_rag,
            }

        # Display final comparison results
        print("\n" + "=" * 60)
        print("📊 FINAL COMPARISON RESULTS")
        print("=" * 60)

        if "long_context" in results:
            lc = results["long_context"]
            print("🔥 Long Context Results:")
            print(f"   ✅ File: {lc['output_file']}")
            print(
                f"   📈 Success: {lc['correct_count']}/{lc['total_questions']} correct ({lc['success_rate']:.1f}%)"
            )
            if isinstance(lc["total_cost"], (int, float)):
                print(f"   💰 Cost: ${lc['total_cost']:.6f}")
            else:
                print(f"   💰 Cost: {lc['total_cost']}")

        if "regular_rag" in results:
            rag = results["regular_rag"]
            print("🔍 Regular RAG Results:")
            print(f"   ✅ File: {rag['output_file']}")
            print(
                f"   📈 Success: {rag['correct_count']}/{rag['total_questions']} correct ({rag['success_rate']:.1f}%)"
            )
            if isinstance(rag["total_cost"], (int, float)):
                print(f"   💰 Cost: ${rag['total_cost']:.6f}")
            else:
                print(f"   💰 Cost: {rag['total_cost']}")

        # Show comparison if both methods were run
        if len(results) == 2:
            lc = results["long_context"]
            rag = results["regular_rag"]

            print("\n🏆 COMPARISON:")
            if lc["success_rate"] > rag["success_rate"]:
                diff = lc["success_rate"] - rag["success_rate"]
                print(
                    f"   🥇 Long Context wins by {diff:.1f}% ({lc['success_rate']:.1f}% vs {rag['success_rate']:.1f}%)"
                )
            elif rag["success_rate"] > lc["success_rate"]:
                diff = rag["success_rate"] - lc["success_rate"]
                print(
                    f"   🥇 Regular RAG wins by {diff:.1f}% ({rag['success_rate']:.1f}% vs {lc['success_rate']:.1f}%)"
                )
            else:
                print(f"   🤝 It's a tie! Both achieved {lc['success_rate']:.1f}%")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
