# RAG PDF Chat: Context-Aware Document Intelligence

**A high-performance Retrieval-Augmented Generation (RAG) application that enables researchers to conduct semantic inquiries and extract nuanced insights from complex PDF datasets in real-time.**

---

## 🚀 Core Features

* **Semantic Document Interrogation:** moves beyond keyword matching to understand the underlying context of research papers and technical manuals.
* **Vectorized Knowledge Retrieval:** utilizes high-dimensional embeddings to pinpoint relevant information across multiple long-form documents.
* **Asynchronous Processing:** implements efficient data ingestion and vector store creation to handle text extraction without UI latency.
* **Stateful Conversational Memory:** maintains context across multiple queries, allowing for iterative deep-dives into specific document sections.
* **Streamlined UI/UX:** provides a dedicated research dashboard built for rapid document processing and intuitive interaction via Streamlit.

## 🛠 Technical Stack

This project leverages a modern AI stack designed for modularity and performance:

* **Orchestration:** [LangChain](https://www.langchain.com/) (Chains, Memory, and Retrieval modules).
* **LLM & Embeddings:** [Google Gemini](https://ai.google.dev/) (specifically `gemini-1.5-flash`) and `models/embedding-001`.
* **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search).
* **Web Framework:** [Streamlit](https://streamlit.io/).
* **Document Processing:** [PyPDF2](https://pypi.org/project/PyPDF2/).

## 🏗 System Architecture

The application follows a rigorous RAG pipeline to ensure data integrity and retrieval accuracy:

1. **Data Ingestion:** Raw PDF data is parsed and converted into a normalized text stream using `PdfReader`.
2. **Recursive Chunking:** Text is split into $1000$ character segments with a $200$ character overlap to maintain semantic continuity across boundaries.
3. **Vectorization:** Each chunk is converted into a high-dimensional vector using Google's generative AI embedding models.
4. **Indexing:** Vectors are stored in a FAISS index for sub-millisecond similarity searching.
5. **Retrieval & Synthesis:** Upon a user query, the `ConversationalRetrievalChain` retrieves relevant chunks and passes them as context to the Gemini LLM for grounded response generation.

---

## 🔬 Scientific & Research Relevance

In a research-intensive environment like **IISc Bangalore**, this tool serves as a force-multiplier for academic productivity:

* **Literature Review Automation:** Rapidly synthesize findings across hundreds of papers to identify research gaps or methodology trends.
* **Technical Manual Synthesis:** Instantly query complex instrumentation manuals for specific calibration parameters or troubleshooting protocols.
* **Grant & Patent Analysis:** Efficiently cross-reference new findings against existing patent databases or grant requirements to ensure novelty.

---

## 💻 Installation & Usage

### Prerequisites

* Python 3.9+
* A Google AI Studio API Key

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/nikamsudarshan/rag-pdf-chat.git
cd rag-pdf-chat

```


2. **Install dependencies:**
```bash
pip install streamlit pypdf2 langchain langchain-google-genai faiss-cpu python-dotenv

```


3. **Configure Environment:**
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY="your_actual_api_key_here"

```


4. **Run the Application:**
```bash
streamlit run app.py

```



---

## 👨‍🔬 About the Author

I am a **Mechatronics Engineering student** at **New Horizon Institute of Technology** (affiliated with the **University of Mumbai**). My academic focus lies at the intersection of robotics, automation, and intelligent systems, supplemented by a **Minor in Information Technology**.

As the founding member of the **ARMORY Robotics Club**, I am passionate about building end-to-end R&D projects—ranging from hexapod kinematics to AI-driven document intelligence. My goal is to leverage my background in hardware-software integration to contribute to cutting-edge research in autonomous systems and applied AI.

