from fastapi import FastAPI
import chatbot
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

intents = json.loads(open('./intents_vn.json', encoding='utf-8').read())


@app.get("/chatbot/")
async def get_response(message: str):
    pred_intent = chatbot.predict_class(message)
    resp = chatbot.get_response(pred_intent, intents)
    return {"response": resp}
