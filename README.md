# PACR-Chatbot Feature

## Project Structure
```bash
pacr-chatbot/
│
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── db.py               # SQLite setup and seed
│   ├── models.py           # SQLAlchemy models (users, chats documents)
│   ├── schemas.py          # Pydantic schemas
│   ├── crud.py             # Controllers / DB functions
│   ├── llm_service.py      # OpenAI API wrapper
│   ├── rag.py              # FAISS index, retrieval + context builder
│   └── routers.py          # FastAPI routes (login via email, chat, upload)
│
├── frontend/
│   ├── app.py              # Streamlit entry point
│   ├── sidebar.py          # Email input, chat history
│   ├── chat.py             # Chat UI (bubbles) + upload
│   └── api_client.py       # Calls FastAPI backend
│
├── data/
│   └── seed_data.json      # Initial data (sample user/email, sample docs)
│
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── .env          (you need to create)
└── chat_app.db   (will be created)
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

5. Initalize and seed db
```bash
cd backend
python db.py
```

6. Run Backend (FastAPI) from project root folder
```bash
uvicorn backend.main:app --reload --port 8000
```

7. Open new terminal and activate venv (dont create again)
```bash
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

8. Run Frontend (Streamlit) from project root folder
```bash
python -m streamlit run frontend/app.py
```

9. Have fun!
