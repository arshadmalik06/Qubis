# IS_Solution
# QuBIS 🇮🇳

**AI-Based Bureau of Indian Standards (BIS) Compliance Assistant & Checklist Generator**

QuBIS is a smart platform designed to simplify BIS compliance for manufacturers, MSMEs, and industries. By combining Retrieval-Augmented Generation (RAG) with strict data pipelines, QuBIS instantly transforms complex government standards into actionable, trackable steps.

## ✨ Core Features

*   **🤖 Streaming AI Assistant:** A conversational interface that answers queries based on real BIS documents. Includes a split-screen PDF viewer that highlights the exact clause the AI used to answer your question.
*   **📋 Dynamic Compliance Checklists:** Input a product and its intended use to generate a strict, personalized compliance roadmap containing required documents, testing actions, and certification steps.
*   **✅ Progress Tracking:** Interactive UI to track task completion (Completed/Pending).
*   **💾 Router-Proof Memory:** Seamlessly saves all chat sessions and checklist states to local storage, surviving page refreshes and navigation.
*   **📄 Export Summaries:** One-click client-side generation of downloadable compliance summary reports.

## 🛠️ Tech Stack

**Frontend**
*   React (TypeScript)
*   Tailwind CSS
*   React Router (SPA Navigation)
*   Browser LocalStorage API (State Persistence)

**Backend & AI**
*   FastAPI (Python REST & SSE Streaming)
*   Pydantic (Strict Data Validation)
*   ChromaDB (Vector Database for RAG)
*   Qwen LLM (Streaming Agent & Structured JSON Generation)

## 🚀 Getting Started

### Prerequisites
*   Node.js (v18+)
*   Python (3.10+)
*   Windows 11 / Linux / macOS

### 1. Backend Setup (FastAPI)
Navigate to the backend directory, set up your virtual environment, and start the server:

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
