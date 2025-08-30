# PACR-Chatbot Feature

## Project Structure
```bash
pacr-chatbot/
│
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── db.py               # SQLite setup and seed
│   └── models.py           # SQLAlchemy models (users, chats documents)
│
├── frontend/
│   └── app.py              # Streamlit entry point
│
├── requirements.txt
├── .env          (you need to create)
└── database.db   (will be created)
```

## Setup
1. Clone repo and chaneg directory
```bash
git clone https://github.com/Chinzzii/PACR-chatbot.git
cd PACR-chatbot
```
2. Create virtual env and activate
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install requirements from root folder
```bash
pip install -r requirements.txt
```

4. Create .env in project root and add key
```bash
OPENAI_API_KEY=sk-xxxx-your-openai-key
API_BASE=http://localhost:8000/api
```

5. Run Backend (FastAPI) from project root folder
```bash
uvicorn backend.main:app --reload --port 8000
```

6. Open new terminal and activate venv (dont create again)
```bash
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

7. Run Frontend (Streamlit) from project root folder
```bash
python streamlit run frontend/app.py
```

8. Have fun!
