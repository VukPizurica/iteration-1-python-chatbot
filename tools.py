import requests
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ---------------- Master Runner ----------------
def run_task_with_llm_decision(task: str) -> str:
    """
    Main entry point:
    1. Ask LLM which tool to use
    2. Call the corresponding function
    3. Return output
    """
    decision = llm_choose_tool(task)
    tool = decision.get("tool", "llm")
    task_to_run = decision.get("task", task)

    if tool == "search":
        return call_wikipedia(task_to_run)
    if tool == "weather":
        return call_weather(task_to_run)
    if tool == "file":
        return call_file_tool(task_to_run)
    return call_llm(task_to_run)


# ---------------- LLM Tool Selector ----------------
def llm_choose_tool(task: str) -> dict:
    # Force "file" if certain keywords are present
    file_keywords = [
        "file",
        "store in a file",
        "save as",
        "generate csv",
        "output json",
        "write to file",
        "create a file",
    ]
    if any(k.lower() in task.lower() for k in file_keywords):
        return {"tool": "file", "task": task}

    # Otherwise, call LLM to decide
    system_prompt = """
    You are an AI assistant that ONLY returns JSON.
    Classify tasks as: "search", "weather", "file", "llm".
    Return JSON with keys: "tool" and "task".
    """
    user_prompt = f"{system_prompt}\n\nUser query: {task}\n\nJSON only:"
    response = call_llm(user_prompt)
    try:
        return json.loads(response)
    except Exception:
        return {"tool": "llm", "task": task}


# ---------------- Tool Implementations ----------------


# --- Wikipedia Search ---
def call_wikipedia(query: str) -> str:
    try:
        base_url = "https://en.wikipedia.org/w/api.php"
        headers = {"User-Agent": "AI-Agent-Project/1.0"}
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        search_res = requests.get(
            base_url, params=search_params, headers=headers, timeout=5
        )
        search_data = search_res.json()
        if not search_data.get("query", {}).get("search"):
            return "[Wikipedia] No results found."

        page_title = search_data["query"]["search"][0]["title"]
        summary_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": page_title,
            "format": "json",
        }
        summary_res = requests.get(
            base_url, params=summary_params, headers=headers, timeout=5
        )
        pages = summary_res.json().get("query", {}).get("pages", {})
        page_id = next(iter(pages))
        extract = pages[page_id].get("extract")
        if extract:
            return f"[Wikipedia - {page_title}]\n{extract[:500]}..."
        return f"[Wikipedia] Page found for '{page_title}' but no summary available."
    except Exception as e:
        return f"Wikipedia Tool Error: {str(e)}"


# --- Weather ---
def call_weather(query: str) -> str:
    try:
        city = query.split("in")[-1].strip() if "in" in query else "London"
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_res = requests.get(geo_url, timeout=5)
        if geo_res.status_code != 200:
            return "Geocoding failed"
        geo_data = geo_res.json()
        if "results" not in geo_data:
            return "City not found"
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_res = requests.get(weather_url, timeout=5)
        if w_res.status_code == 200:
            current = w_res.json().get("current_weather", {})
            temp = current.get("temperature")
            wind = current.get("windspeed")
            return (
                f"[Weather API]\nCity: {city}\nTemperature: {temp}°C\nWind: {wind} km/h"
            )
        return f"Weather request failed: {w_res.status_code}"
    except Exception as e:
        return f"Weather error: {e}"


# --- File Tool ---
FILES_FOLDER = Path("saved_files")
FILES_FOLDER.mkdir(exist_ok=True)


def safe_json_parse(text: str) -> dict:
    """
    Extract JSON object from text and parse it.
    Works even if LLM adds extra text or code fences.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def slugify_filename_with_extension(name: str) -> str:
    """
    Slugify the filename while preserving the extension:
    - Lowercase
    - Replace spaces with underscores
    - Remove invalid chars
    - Keep the extension at the end
    """
    name = name.strip()
    # Split extension
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        ext = "." + ext.lower()
    else:
        stem, ext = name, ".txt"  # default to txt if missing
    # Clean stem
    stem = stem.lower()
    stem = re.sub(r"[^a-z0-9\s_-]", "", stem)
    stem = re.sub(r"\s+", "_", stem)
    return stem + ext


def call_file_tool(task: str) -> str:
    """
    Ask LLM to generate a file in strict JSON format and save it locally.
    LLM chooses the filename AND the most appropriate extension.
    """
    prompt = f"""
    You are an AI that generates content to store in files.
    The JSON MUST escape all newlines and quotes inside the "content" string
    so that it is valid JSON. Example: "line1\nline2" not raw line breaks.
    RETURN ONLY JSON with EXACT keys:
    {{
        "filename": "descriptive_filename.ext",
        "content": "The content of the file in natural readable format with proper sentences and paragraphs."
    }}
    Requirements:
    1. The 'filename' key must contain a descriptive filename in 3-6 words.
       - Use underscores instead of spaces
       - Lowercase letters only
       - Only alphanumeric characters and underscores
    2. The 'content' key must contain natural readable text.
       - Do NOT replace spaces with underscores
       - Preserve punctuation, capitalization, and paragraphs
    3. Use the most appropriate extension for the content (.txt, .md, .py, .json, etc.)
    4. Do NOT include any explanations, code fences, or extra text.
    Task: {task}
    """
    response = call_llm(prompt)
    try:
        data = safe_json_parse(response)
        raw_filename = data["filename"]
        content = data["content"]

        # Slugify filename only
        file_path = FILES_FOLDER / slugify_filename_with_extension(raw_filename)

        # Handle duplicates
        counter = 1
        while file_path.exists():
            stem, ext = file_path.stem, file_path.suffix
            file_path = FILES_FOLDER / f"{stem}_{counter}{ext}"
            counter += 1

        # Save file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[File Tool] Saved '{file_path.name}' successfully."
    except Exception as e:
        return f"File Tool error: {e}\nRaw LLM response: {response}"


# ---------------- LLM Call Wrappers ----------------
def call_llm(prompt: str) -> str:
    """Try multiple LLM providers until one succeeds."""
    for fn in [call_hf_model, call_groq, call_gemini]:
        result = fn(prompt)
        # print(f"Result returned from the LLM: {result}")clear
        
        if result:
            return result
    return "All LLM providers failed."


def call_hf_model(prompt: str) -> str:
    for model in MODEL_LIST:
        try:
            r = requests.post(
                BASE_CHAT,
                headers=HEADERS,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if r.status_code == 200:
                text = (
                    r.json().get("choices", [{}])[0].get("message", {}).get("content")
                )
                if text:
                    return text
        except Exception as e:
            print(f"{model} error: {e}")
    return None


def call_groq(prompt: str) -> str:
    for model in GROQ_MODELS:
        try:
            resp = groq_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"{model} error: {e}")
    return None


def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Gemini error: {e}")
    return None


# ---------------- Keywords / Constants ----------------
SEARCH_KEYWORDS = [
    "search",
    "find",
    "lookup",
    "look up",
    "who is",
    "what is",
    "where is",
    "info on",
    "details on",
    "tell me about",
]

WEATHER_KEYWORDS = [
    "weather",
    "temperature",
    "forecast",
    "rain",
    "sunny",
    "wind",
    "climate",
]

# ---------------- Model Definitions ----------------
HF_TOKEN = os.getenv("HF_API_KEY")
BASE_CHAT = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
MODEL_LIST = [
    "deepseek-ai/DeepSeek-R1",
    "openai/gpt-oss-120b:fastest",
    "openai/gpt-oss-120b:cheapest",
]

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)
GROQ_MODELS = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
