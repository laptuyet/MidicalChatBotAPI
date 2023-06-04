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


def getSpO2Type(spo2):
    if 95 <= spo2 <= 100:
        return 'normal_spo2'
    elif 90 <= spo2 < 95:
        return 'warn_spo2'
    else:
        return 'low_spo2'


@app.get("/chatbot/")
async def get_response(message: str):
    flag = message.__contains__('BPM') and message.__contains__('SPO2')
    if flag:
        txts = message.split(' ')
        heart_rate = int(txts[0][4:])
        spo2 = int(txts[1][5:])

        hr_message = getHeartRateType(heart_rate)
        spo2_message = getSpO2Type(spo2)

        pred_intents = chatbot.predict_class(hr_message)
        hr_res = chatbot.get_response(pred_intents, intents)
        pred_intents = chatbot.predict_class(spo2_message)
        spo2_res = chatbot.get_response(pred_intents, intents)

        resp = hr_res + '\n' + spo2_res
    else:
        pred_intent = chatbot.predict_class(message)
        resp = chatbot.get_response(pred_intent, intents)
    return {"response": resp}
