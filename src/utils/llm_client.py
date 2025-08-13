"""LLM client for Gemini 2.5 Flash."""

import re
import time
from typing import Any, Dict, List

import google.generativeai as genai

from .config import config


class GeminiClient:
    """Client for interacting with Google's Gemini 2.5 Flash model."""

    def __init__(self):
        """Initialize the Gemini client."""
        genai.configure(api_key=config.google_api_key)

        # Disable ALL safety filters for research purposes
        # Using the most permissive settings possible
        safety_settings = [
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
            },
        ]

        # Try creating model with no safety settings first
        try:
            self.model = genai.GenerativeModel(config.llm_model, safety_settings=None)
        except Exception:
            # Fallback to explicit BLOCK_NONE settings
            self.model = genai.GenerativeModel(
                config.llm_model, safety_settings=safety_settings
            )

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response using Gemini 2.5 Flash.

        Args:
            prompt: The input prompt

        Returns:
            Dictionary containing response text, token usage, and timing info
        """
        start_time = time.time()

        try:
            # Generate content with explicit safety settings override and detailed safety feedback
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.llm_temperature,
                    max_output_tokens=1000,  # Sufficient for multiple choice answers
                ),
                safety_settings=[
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                    },
                ],
            )

            end_time = time.time()

            # Check if the response was blocked
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                # Check finish reason (2 = SAFETY according to the API docs)
                if candidate.finish_reason == 2:
                    # Response was blocked by safety filters
                    safety_categories = []
                    if (
                        hasattr(candidate, "safety_ratings")
                        and candidate.safety_ratings
                    ):
                        try:
                            for rating in candidate.safety_ratings:
                                safety_categories.append(
                                    {
                                        "category": str(
                                            getattr(rating, "category", "")
                                        ),
                                        "prob": getattr(rating, "probability", None),
                                        "blocked": getattr(rating, "blocked", None),
                                    }
                                )
                        except Exception:
                            pass
                    # Retry once with a neutralized prompt to reduce trigger surface
                    neutralized_prompt = self._neutralize_prompt(prompt)
                    retry_start = time.time()
                    retry_response = self.model.generate_content(
                        neutralized_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=config.llm_temperature,
                            max_output_tokens=1000,
                        ),
                        safety_settings=[
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                            },
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                            },
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                            },
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                            },
                        ],
                    )

                    retry_end = time.time()
                    retry_candidate = (
                        retry_response.candidates[0]
                        if retry_response.candidates
                        else None
                    )
                    if (
                        retry_candidate
                        and getattr(retry_candidate, "finish_reason", None) != 2
                    ):
                        retry_text = (
                            retry_response.text
                            if hasattr(retry_response, "text")
                            else ""
                        )
                        input_tokens = int(len(neutralized_prompt.split()) * 1.3)
                        output_tokens = int(len(retry_text.split()) * 1.3)
                        return {
                            "response": retry_text,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "response_time": (retry_end - retry_start),
                            "success": True,
                            "note": "neutralized_prompt_retry",
                            "safety_ratings_first_attempt": safety_categories or None,
                        }

                    return {
                        "response": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "response_time": end_time - start_time,
                        "success": False,
                        "error": f"Content blocked by safety filters. Finish reason: {candidate.finish_reason}",
                        "safety_ratings": safety_categories or None,
                    }

            # Extract response text
            response_text = response.text if hasattr(response, "text") else ""

            # Calculate token usage (approximate)
            input_tokens = len(prompt.split()) * 1.3  # Rough approximation
            output_tokens = len(response_text.split()) * 1.3

            return {
                "response": response_text,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(input_tokens + output_tokens),
                "response_time": end_time - start_time,
                "success": True,
            }

        except Exception as e:
            end_time = time.time()
            return {
                "response": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e),
            }

    def _neutralize_prompt(self, prompt: str) -> str:
        """Replace likely safety-triggering terms with neutral placeholders to reduce blocks.

        Keeps structure intact while redacting sensitive tokens.
        """
        sensitive_terms = {
            # military / weapons
            "weapon",
            "weapons",
            "military",
            "army",
            "navy",
            "air force",
            "missile",
            "missiles",
            "bomb",
            "bombs",
            "gun",
            "guns",
            "rifle",
            "rifles",
            "pistol",
            "pistols",
            "grenade",
            "grenades",
            "war",
            "warfare",
            "combat",
            "drone",
            "drones",
            "uav",
            "icbm",
            "torpedo",
            "torpedoes",
            # general violence
            "kill",
            "killing",
            "murder",
            "assault",
            "attack",
            "attacks",
            "explosive",
            "explosives",
        }

        def redact(match: re.Match) -> str:
            word = match.group(0)
            return "[redacted]"

        pattern = re.compile(
            r"("
            + "|".join(sorted(map(re.escape, sensitive_terms), key=len, reverse=True))
            + r")",
            re.IGNORECASE,
        )
        return pattern.sub(redact, prompt)

    def create_multiple_choice_prompt(
        self, question: str, options: List[str], context: str
    ) -> str:
        """
        Create a structured prompt for multiple choice questions.

        Args:
            question: The question to answer
            options: List of answer options
            context: Relevant context/documents

        Returns:
            Formatted prompt string
        """
        options_text = "\n".join([f"{i}. {option}" for i, option in enumerate(options)])

        prompt = f"""You are an expert at reading comprehension. Based on the provided context, answer the multiple choice question.

CONTEXT:
{context}

QUESTION: {question}

OPTIONS:
{options_text}

INSTRUCTIONS:
- Read the context carefully
- Select the most accurate answer based ONLY on the information provided in the context
- Respond with just the number (0, 1, 2, or 3) of the correct answer
- If the context doesn't contain enough information, make your best guess based on what's available

ANSWER: """

        return prompt

    def create_open_ended_prompt(self, question: str, context: str) -> str:
        """
        Create a structured prompt for open-ended questions.

        Args:
            question: The question to answer
            context: Relevant context/documents

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert at reading comprehension and question answering. Based on the provided context, answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Read the context carefully
- Answer based ONLY on the information provided in the context
- Be precise and concise in your response
- If the question asks for a specific fact (number, date, name), provide just that information
- If the context doesn't contain the answer, respond with "I cannot find this information in the provided context"

ANSWER: """

        return prompt
