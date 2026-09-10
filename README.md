# 🩺 AI-MEDIBOT — AI-Powered Medical Chatbot

> A Retrieval-Augmented Generation (RAG) based medical chatbot that uses trusted medical documents to provide context-aware answers to healthcare-related queries.

**AI-MEDIBOT** is an AI-powered medical question-answering system built with **Python, LangChain, Hugging Face, FAISS, Streamlit, and Groq**.

Instead of relying solely on an LLM's internal knowledge, the system retrieves relevant information from a curated medical knowledge base and provides it as context to the language model before generating a response.

> ⚠️ **Disclaimer:** AI-MEDIBOT is an educational/research project and is **not a substitute for professional medical advice, diagnosis, or treatment.** Always consult a qualified healthcare professional for medical concerns.

---

## ✨ Features

* 🧠 **Retrieval-Augmented Generation (RAG)**

  * Retrieves relevant information from medical documents before generating an answer.
* 🔎 **Semantic Search**

  * Uses vector embeddings to find documents relevant to the user's query.
* 🤗 **Hugging Face Embeddings**

  * Uses `sentence-transformers/all-MiniLM-L6-v2` for document and query embeddings.
* 🗂️ **FAISS Vector Database**

  * Efficient similarity search over the medical knowledge base.
* 🔗 **LangChain Pipeline**

  * Handles document processing, retrieval, prompting, and LLM integration.
* ⚡ **Groq LLM Inference**

  * Uses the `openai/gpt-oss-20b` model through Groq for fast response generation.
* 💬 **Interactive Streamlit UI**

  * Provides a simple conversational interface.
* 📚 **PDF Knowledge Base**

  * Medical PDFs can be processed and converted into searchable vector embeddings.
* 🎯 **Context-Grounded Responses**

  * The chatbot is instructed to answer using the retrieved context rather than inventing information.

---

## 🏗️ Architecture

```text
                  ┌──────────────────────┐
                  │   Medical PDF Files  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    PDF Loading       │
                  │     PyPDFLoader      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Text Chunking      │
                  │ Recursive Splitter   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Hugging Face         │
                  │ Embeddings           │
                  │ MiniLM-L6-v2         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    FAISS Vector      │
                  │       Store          │
                  └──────────┬───────────┘
                             │
                       User Question
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Semantic Retrieval   │
                  │       Top-K = 3      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Context + Query    │
                  │   Custom Prompt      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      Groq LLM        │
                  │  openai/gpt-oss-20b  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Medical Response   │
                  └──────────────────────┘
```

---

## 🛠️ Tech Stack

| Technology                | Purpose                             |
| ------------------------- | ----------------------------------- |
| **Python**                | Core programming language           |
| **LangChain**             | RAG and LLM orchestration           |
| **Hugging Face**          | Text embeddings and LLM integration |
| **FAISS**                 | Vector similarity search            |
| **Groq**                  | Fast LLM inference                  |
| **Streamlit**             | Web-based chatbot interface         |
| **PyPDF**                 | Medical PDF processing              |
| **Sentence Transformers** | Semantic embeddings                 |
| **python-dotenv**         | Environment variable management     |

---

## 📁 Project Structure

```text
AI-MEDIBOT/
│
├── app.py
│   └── Streamlit chatbot application
│
├── memory_for_llm.py
│   └── Loads PDFs, creates chunks, generates embeddings
│       and builds the FAISS vector database
│
├── connect_memo_with_llm.py
│   └── Connects the FAISS knowledge base with
│       a Hugging Face LLM using LangChain
│
├── requirements.txt
│   └── Python dependencies
│
├── data/
│   └── Medical PDF documents
│
├── vectorstore/
│   └── FAISS vector database
│
└── .env
    └── API credentials and environment variables
```

The repository currently contains the Streamlit application, PDF-to-vector pipeline, LLM connection pipeline, and dependency configuration.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/Krizhere/AI-MEDIBOT.git
cd AI-MEDIBOT
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project dependencies include Streamlit, LangChain, LangChain-HuggingFace, LangChain-Groq, FAISS, PyPDF, Sentence Transformers, and related packages.

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_ACCESS_TOKEN=your_huggingface_token
```

### API Keys

You will need:

* **Groq API Key** — used by the Streamlit application for LLM inference.
* **Hugging Face Access Token** — used by the Hugging Face-based inference pipeline.

Never commit your `.env` file or expose API keys publicly.

---

# 📚 Building the Knowledge Base

Place your medical PDF documents inside:

```text
data/
```

Then run:

```bash
python memory_for_llm.py
```

The pipeline performs the following steps:

```text
Medical PDFs
     ↓
PDF Loading
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Hugging Face Embeddings
     ↓
FAISS Vector Database
```

The current implementation uses:

```text
Chunk Size: 500
Chunk Overlap: 50
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
```

The resulting FAISS database is saved to:

```text
vectorstore/db_faiss
```

These settings correspond to the current repository implementation.

---

# 💬 Running the Chatbot

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the URL displayed by Streamlit, typically:

```text
http://localhost:8501
```

You can then enter medical questions through the chat interface.

---

# 🔄 How RAG Works in AI-MEDIBOT

AI-MEDIBOT follows a Retrieval-Augmented Generation workflow.

### 1. User asks a question

```text
"What are the symptoms of diabetes?"
```

### 2. Query is converted into an embedding

The Hugging Face `all-MiniLM-L6-v2` model converts the question into a numerical vector.

### 3. FAISS searches the knowledge base

The system searches the FAISS vector database for the most semantically relevant chunks.

The current application retrieves the top **3** results.

### 4. Retrieved context is provided to the LLM

The retrieved medical information and user's question are inserted into a custom prompt.

### 5. LLM generates the answer

The application sends the grounded prompt to:

```text
openai/gpt-oss-20b
```

through Groq.

### 6. Response is displayed

The generated answer is displayed through the Streamlit chat interface.

---

# 🧠 Prompt Grounding

The chatbot uses a context-restricted prompting strategy:

```text
Use the pieces of information provided in the context
to answer user's question.

If you don't know the answer, say that you don't know.
Don't try to make up an answer.

Don't provide anything outside of the given context.
```

This helps reduce unsupported answers by instructing the model to stay grounded in retrieved medical information.

---

# 🤖 LLM Configuration

The Streamlit version currently uses:

```python
ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    groq_api_key=os.environ["GROQ_API_KEY"]
)
```

The temperature is set to `0.0` to encourage more deterministic responses.

The repository also includes a separate Hugging Face implementation using:

```text
mistralai/Mistral-7B-Instruct-v0.2
```

through `HuggingFaceEndpoint`.

---

# 📌 Example Queries

You can ask questions such as:

```text
What are the common symptoms of anemia?

What are the symptoms of diabetes?

What causes high blood pressure?

What are the common symptoms of asthma?

What are the risk factors for heart disease?
```

The chatbot retrieves relevant information from the indexed medical documents before generating its response.

---

# 🎯 Project Goals

The main goals of AI-MEDIBOT are:

* Build a practical **RAG application**
* Explore **LangChain-based LLM pipelines**
* Implement **semantic search with FAISS**
* Use **Hugging Face embeddings**
* Build an interactive AI application using **Streamlit**
* Reduce hallucination by grounding responses in retrieved documents
* Understand the complete workflow from **documents → embeddings → retrieval → LLM → response**

---

# 🔮 Future Improvements

Potential improvements include:

* [ ] Display retrieved source documents to users
* [ ] Add document/source citations to every response
* [ ] Improve medical-domain document filtering
* [ ] Add conversation memory
* [ ] Add multilingual support
* [ ] Add voice input/output
* [ ] Add medical image analysis
* [ ] Add authentication and user profiles
* [ ] Add evaluation metrics for RAG accuracy
* [ ] Implement hybrid keyword + semantic search
* [ ] Add reranking for retrieved documents
* [ ] Deploy the application to a cloud platform
* [ ] Add automated tests and CI/CD
* [ ] Improve prompt safety and medical refusal handling

---

# ⚠️ Medical Disclaimer

AI-MEDIBOT is intended for **educational and experimental purposes only**.

It should **not** be used to:

* Diagnose medical conditions
* Prescribe medication
* Replace a doctor or healthcare professional
* Make emergency medical decisions
* Determine an individual's treatment plan

The information generated by an AI system may be incomplete, incorrect, or unsuitable for a particular patient.

**For medical emergencies or serious health concerns, consult a qualified healthcare professional or appropriate emergency service.**

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/your-feature
```

3. Make your changes
4. Commit your changes

```bash
git commit -m "Add: your feature"
```

5. Push the branch

```bash
git push origin feature/your-feature
```

6. Open a Pull Request

---

# 📜 License

This project is currently provided for educational and research purposes.

If you intend to distribute or deploy the project, consider adding an appropriate open-source license such as **MIT**.

---

# 👨‍💻 Author

**Krish Kumar**

* GitHub: [@Krizhere](https://github.com/Krizhere)
* Project: [AI-MEDIBOT](https://github.com/Krizhere/AI-MEDIBOT)

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub!

```text
AI-MEDIBOT
├── RAG
├── LangChain
├── Hugging Face
├── FAISS
├── Groq
├── Streamlit
└── Medical Knowledge Base
```
