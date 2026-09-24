# YouTube RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) application that answers questions about a YouTube video using its transcript.

The application fetches the video's transcript, splits it into smaller chunks, converts the chunks into vector embeddings, and retrieves the most relevant context before generating an answer with Llama 3.1.

## Features

- Fetches English or Hindi YouTube transcripts
- Splits transcripts into overlapping text chunks
- Uses `all-MiniLM-L6-v2` for semantic embeddings
- Uses FAISS for similarity search
- Generates grounded answers with `meta-llama/Llama-3.1-8B-Instruct`
- Answers only from the retrieved transcript context

## Tech Stack

- Python
- LangChain
- Hugging Face
- FAISS
- Sentence Transformers
- YouTube Transcript API

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Create and activate a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Hugging Face

Create a `.env` file in the project root and add your Hugging Face access token:

```env
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
```

Make sure your token has permission to use the selected Hugging Face model.

### 5. Run the application

```bash
python YT_Chatbot.py
```

The current video is configured in `YT_Chatbot.py` using its YouTube video ID. Update the `video_id` value to analyze another video.

## How It Works

1. Retrieves the selected video's transcript from YouTube.
2. Splits the transcript into chunks with overlapping context.
3. Creates vector embeddings for each chunk.
4. Finds the three most relevant chunks for a question.
5. Sends the retrieved context to the language model.
6. Prints an answer grounded in the transcript.

## Example

```text
Question: What is discussed in this video?

Answer: The video discusses the exploration of the tree of knowledge and the potential use of AI to design tools, run simulations, and expand human understanding.
```

## Notes

- The video must have an available English or Hindi transcript.
- The Hugging Face model may require access approval and a valid API token.
- The first run may download the sentence-transformer model locally.

## License

This project is intended for learning and experimentation with Retrieval-Augmented Generation.
