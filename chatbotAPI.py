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


def getHeartRateType(heart_rate):
    if heart_rate > 100:
        return 'high_bpm'
    elif 65 <= heart_rate <= 100:
        return 'normal_bpm'
    else:
        return 'low_bpm'


@app.get("/chatbot/")
async def get_response(message: str):
    flag = message.__contains__('BPM') and message.__contains__('SPO2')
    if flag:
        txts = message.split(' ')
        heart_rate = int(txts[0][4:])
        message = getHeartRateType(heart_rate)

    pred_intent = chatbot.predict_class(message)
    resp = chatbot.get_response(pred_intent, intents)
    return {"response": resp}
