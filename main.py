
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq, APIError, APIConnectionError
from pydantic import BaseModel

# Load environment variables from backend/.env
load_dotenv()

app = FastAPI(title="Web Search Agent API")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"status": "Backend is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("GROQ_API_KEY")

    # Check API key
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY missing. Check your backend .env file."
        )

    # Check empty message
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a message."
        )

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Answer in simple, clear language."
                    ),
                },
                {
                    "role": "user",
                    "content": request.message,
                },
            ],
            max_completion_tokens=500,
        )

        answer = response.choices[0].message.content

        return {
            "reply": answer or "The AI returned an empty response."
        }

    except APIConnectionError as error:
        print("CONNECTION ERROR:", repr(error))
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Groq. Check your internet connection."
        )

    except APIError as error:
        # Print the actual error in the backend terminal
        print("GROQ API ERROR TYPE:", type(error).__name__)
        print("GROQ API ERROR:", str(error))

        raise HTTPException(
            status_code=502,
            detail=f"Groq API error: {type(error).__name__}. Check backend terminal."
        )

    except Exception as error:
        print("UNEXPECTED ERROR TYPE:", type(error).__name__)
        print("UNEXPECTED ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {type(error).__name__}"
        )
