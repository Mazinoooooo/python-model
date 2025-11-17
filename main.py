import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY:", GEMINI_API_KEY)

# Load trained model and encoders
model = joblib.load("cousin_group_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
target_le = joblib.load("target_encoder.joblib")
model_columns = joblib.load("model_columns.joblib")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://capstone-195b7.web.app", "https://capstone-195b7.firebaseapp.com"],  # replace with your deployed frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User profile model
class UserProfile(BaseModel):
    SHS_Strand: str
    SchoolType: str
    Personality: List[str]
    Workstyle: List[str]
    Interests: List[str]
    Skills: Optional[List[str]] = []
    CourseDuration: Optional[str] = None

# Transform user input for ML model
def transform_user_input(user: UserProfile):
    skill_subject_map = {
        "Math": ["Great in Math", "Good in Math"],
        "Science": ["Great in Science", "Good in Science"],
        "English": ["Great in English", "Good in English"],
        "Filipino": ["Great in Filipino", "Good in Filipino"],
        "SocialStudies": ["Great in Social Studies", "Good in Social Studies"],
        "ICT": ["Great in ICT", "Good in ICT", "Good in ICT/Tech-Voc", "Great in ICT/Tech-Voc"]
    }
    ratings = {subject: 1 for subject in skill_subject_map.keys()}

    for skill in user.Skills:
        for subject, skill_keywords in skill_subject_map.items():
            if skill in skill_keywords:
                if "Great" in skill:
                    ratings[subject] = max(ratings[subject], 4)
                elif "Good" in skill:
                    ratings[subject] = max(ratings[subject], 2)

    def safe_trait(lst, idx):
        return lst[idx] if idx < len(lst) else "None"

    data = {
        "SHS_Strand": [user.SHS_Strand],
        "SchoolType": [user.SchoolType],
        "Math": [ratings["Math"]],
        "Science": [ratings["Science"]],
        "English": [ratings["English"]],
        "Filipino": [ratings["Filipino"]],
        "SocialStudies": [ratings["SocialStudies"]],
        "ICT": [ratings["ICT"]],
        "Personality1": [safe_trait(user.Personality, 0)],
        "Personality2": [safe_trait(user.Personality, 1)],
        "Interest1": [safe_trait(user.Interests, 0)],
        "Interest2": [safe_trait(user.Interests, 1)],
        "Workstyle1": [safe_trait(user.Workstyle, 0)],
        "Workstyle2": [safe_trait(user.Workstyle, 1)],
        "Skill1": [safe_trait(user.Skills, 0)],
        "Skill2": [safe_trait(user.Skills, 1)],
        "Skill3": [safe_trait(user.Skills, 2)],
        "CourseDuration": [user.CourseDuration if user.CourseDuration else "None"]
    }

    return pd.DataFrame(data)

# ML prediction endpoint
@app.post("/predict/")
def predict(user: UserProfile):
    input_df = transform_user_input(user)

    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else "None")
            input_df[col] = le.transform(input_df[col])

    input_df = input_df.reindex(columns=model_columns)

    pred_code = model.predict(input_df)[0]
    pred_group = target_le.inverse_transform([pred_code])[0]

    probs = model.predict_proba(input_df)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = {target_le.inverse_transform([i])[0]: float(probs[i]) for i in top3_idx}

    return {"predicted_group": pred_group, "top_3_groups": top3}

# Async Gemini endpoint
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_content(req: PromptRequest):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": req.prompt}]}]}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

@app.get("/")
def root():
    return {"status": "FastAPI is running!"}
