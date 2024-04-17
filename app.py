from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()

PORT = int(os.get('PORT', 8000))
HOST = '0.0.0.0'

# Load the dataset
df = pd.read_csv("diabetes.csv")

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Initialize and train the model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# FastAPI app initialization
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Define request body model for POST method
class UserReport(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Define routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_diabetes(user_data: UserReport):
    # Make prediction using the model
    user_data_dict = user_data.dict()
    user_data_df = pd.DataFrame(user_data_dict, index=[0])

    user_result = rf.predict(user_data_df)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100

    # Determine output
    if user_result[0] == 0:
        output = 'You are not Diabetic'
    else:
        output = 'You are Diabetic'

    return {"output": output, "accuracy": accuracy}


if __name__ == '__main__':
    uvicorn.run('app.api:app', host = HOST, port = PORT, reload = True)
