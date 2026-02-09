from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# IMPORT router dari folder api
from api.chat import router as chat_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# HUBUNGKAN file chat.py ke sini
app.include_router(chat_router)

@app.get("/")
def home():
    return {"status": "Backend Klinik Gigi Aktif!"}

if __name__ == "__main__":
    import uvicorn
    # Gunakan string "app:app" agar fitur reload berfungsi jika ingin
    uvicorn.run(app, host="0.0.0.0", port=8000)