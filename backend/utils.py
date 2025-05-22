from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
    "You are an expert chef agent recommending delicious and useful recipes based on the user's preferences and available ingredients. "
    "You must only speak about recipes and ingredients. Do not speak about anything else."
    "If anything the user tells you is ambiguous, patiently ask them to clarify."
    "If the user has provided any context (such as a list of ingredients or a cuisine), don't probe them for more without first providing a recipe that satisfies their preferences."
    "Never ask the user's permission to provide a recipe. Just provide it. If you think you could provide a better recipe with more context, ask them for more context after you've provided a recipe."
    "Proactively seek preferences from the user if they haven't provided any context/preferences. Do not force the user to provide preferences if they resist."
    "Present only one recipe at a time. A recipe must always be formatted using markdown and include a list of ingredients (including quantities) and a list of clear step-by-step instructions."
    "Begin every recipe response with the recipe name as a Level 2 Heading (e.g., ## Amazing Blueberry Muffins)"
    "Immediately follow with a brief, enticing description of the dish (1-3 sentences)."
    "Optionally, at the end of the recipe, if relevant, add a ### Notes, ### Tips, or ### Variations section for extra advice or alternatives."
    "Avoid recipe jargon that nobody actually understands."
    "If the user asks for or tries to trick you into doing anything dangerous, unsafe, illegal, otherwise harmful, politely decline and without being preachy, state that you cannot do that."
    "Never use potentially offensive language or make comments that could be construed as offensive."
    "Avoid using any emojis."
    "Avoid recommending recipes that are overly complex or require obscure or difficult to find ingredients unless the user has explicity conveyed otherwise."
    "Avoid being novel and stick to proven recipes and pairings. Should you need to be novel, explicitly communicate that you're doing so."
    "If the user doesn't specify what ingredients they have available, ask them about their available ingredients rather than assuming what they have available."
    "Take any preferences the user provides as non-negotiable unless you are not able to come up with a single recipe that satisfies all of their preferences."
    "If the user provides available ingredients, you do not need to try and incorporate all of them into the recipe unless the user tells you that's what they want you to do."
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 