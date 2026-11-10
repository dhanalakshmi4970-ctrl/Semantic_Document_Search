
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import chromadb
import nbformat

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "dataset"
DOCS_DIR = DATASET / "documents"
DB_DIR = ROOT / "chroma_db"
NOTEBOOK_DIR = ROOT / "notebook"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "week12_semantic_search"


# ==========================================
# 2. EDUCATION DOCUMENTS (10 DOCUMENTS)
# ==========================================

DOCUMENTS = [
    {
        "id": "doc01",
        "title": "Project Based Learning",
        "category": "Teaching Methods",
        "content": "Project based learning helps students learn by investigating real world problems. Students research, collaborate, develop projects and present their findings. Teachers guide students through milestones and assessments."
    },
    {
        "id": "doc02",
        "title": "Flipped Classroom",
        "category": "Teaching Methods",
        "content": "A flipped classroom allows students to study lesson materials at home using videos and readings. Classroom time is used for discussions, problem solving, practical activities and teacher support."
    },
    {
        "id": "doc03",
        "title": "Accessible Online Learning",
        "category": "Educational Technology",
        "content": "Accessible online learning provides digital educational resources for students with different learning needs. Captions, screen readers, alternative text and readable documents improve accessibility."
    },
    {
        "id": "doc04",
        "title": "Cyber Safety for Students",
        "category": "Digital Education",
        "content": "Cyber safety teaches students to protect personal information online. Students should use strong passwords, avoid suspicious links, protect private information and report cyberbullying."
    },
    {
        "id": "doc05",
        "title": "Peer Feedback in Writing",
        "category": "Assessment",
        "content": "Peer feedback allows students to review classmates' writing. Students provide suggestions, identify strengths and recommend improvements. Feedback helps improve writing skills and communication."
    },
    {
        "id": "doc06",
        "title": "Positive Classroom Behaviour",
        "category": "Student Development",
        "content": "Positive classroom behaviour creates a safe learning environment. Teachers establish classroom rules, encourage cooperation, reward positive actions and support students when problems occur."
    },
    {
        "id": "doc07",
        "title": "Information Literacy",
        "category": "Academic Skills",
        "content": "Information literacy helps students find and evaluate reliable information. Students examine sources, verify facts, identify bias, use research databases and cite references correctly."
    },
    {
        "id": "doc08",
        "title": "Personalised Learning Plans",
        "category": "Inclusive Education",
        "content": "Personalised learning plans identify individual student goals, strengths and learning needs. Teachers provide flexible teaching strategies, adapted materials and regular progress assessments."
    },
    {
        "id": "doc09",
        "title": "Blended Learning",
        "category": "Educational Technology",
        "content": "Blended learning combines traditional classroom teaching with online learning activities. Students use digital resources, online quizzes, classroom discussions and practical exercises."
    },
    {
        "id": "doc10",
        "title": "Critical Thinking Skills",
        "category": "Learning Skills",
        "content": "Critical thinking involves analysing information, evaluating evidence and solving problems. Students develop these skills through debates, discussions, case studies and research activities."
    }
]


# ==========================================
# 3. CREATE 15 USER QUERIES
# ==========================================

QUERIES = [
    "How can students learn through real world projects?",
    "What should students do before attending a flipped class?",
    "How can online lessons support students with disabilities?",
    "How can students protect their personal information online?",
    "How can classmates improve each other's writing?",
    "How can teachers encourage good classroom behaviour?",
    "How can students identify trustworthy information?",
    "What information should an individual learning plan contain?",
    "How can teachers combine classroom and online learning?",
    "What activities improve students' critical thinking?",
    "How can students avoid plagiarism in assignments?",
    "What tools help visually impaired students access online lessons?",
    "Why are project deadlines useful for students?",
    "How should students respond to online bullying?",
    "How can teachers support students with different learning speeds?"
]


# ==========================================
# 4. CREATE DATASET FILES
# ==========================================

def create_dataset():

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    for doc in DOCUMENTS:

        file_path = DOCS_DIR / f"{doc['id']}.txt"

        content = (
            f"Title: {doc['title']}\n"
            f"Category: {doc['category']}\n\n"
            f"{doc['content']}"
        )

        file_path.write_text(content, encoding="utf-8")

    with open(DATASET / "queries.csv", "w",
              newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["query_id", "query"])

        for i, query in enumerate(QUERIES, 1):
            writer.writerow([f"query{i:02d}", query])

    print("10 documents created.")
    print("15 queries created.")


# ==========================================
# 5. GENERATE VECTOR EMBEDDINGS
# ==========================================

def generate_embeddings():

    print("\nLoading Sentence Transformer model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    documents = []

    for doc in DOCUMENTS:

        documents.append(
            Document(
                page_content=doc["content"],
                metadata={
                    "document_id": doc["id"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "source": f"dataset/documents/{doc['id']}.txt"
                }
            )
        )

    # Independent Week 12 ChromaDB collection
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(DB_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Add missing documents without duplicating them
    existing_ids = set(vector_store.get()["ids"])

    new_documents = [
        doc for doc in documents
        if doc.metadata["document_id"] not in existing_ids
    ]

    if new_documents:

        vector_store.add_documents(
            documents=new_documents,
            ids=[
                doc.metadata["document_id"]
                for doc in new_documents
            ]
        )

    print("Embeddings generated and stored in ChromaDB.")
    print("Total database documents:", vector_store._collection.count())

    return vector_store, embedding_model


# ==========================================
# 6. SEMANTIC AND KEYWORD SEARCH
# ==========================================

def perform_search(vector_store, embedding_model):

    semantic_results = []
    keyword_results = []
    similarity_results = []

    document_texts = [doc["content"] for doc in DOCUMENTS]

    # TF-IDF keyword search
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(document_texts)

    for index, query in enumerate(QUERIES, 1):

        query_id = f"query{index:02d}"

        print(f"\nSearching query {index}: {query}")

        # ----------------------------------
        # SEMANTIC SEARCH USING LANGCHAIN
        # ----------------------------------

        retrieved_docs = vector_store.similarity_search(
            query,
            k=5
        )

        query_embedding = np.array(
            embedding_model.embed_query(query)
        ).reshape(1, -1)

        retrieved_embeddings = np.array(
            embedding_model.embed_documents(
                [doc.page_content for doc in retrieved_docs]
            )
        )

        scores = cosine_similarity(
            query_embedding,
            retrieved_embeddings
        )[0]

        for rank, (doc, score) in enumerate(
            zip(retrieved_docs, scores), 1
        ):

            metadata = doc.metadata

            row = {
                "query_id": query_id,
                "query": query,
                "method": "Semantic Search",
                "rank": rank,
                "document_id": metadata["document_id"],
                "title": metadata["title"],
                "category": metadata["category"],
                "source": metadata["source"],
                "similarity_score": round(float(score), 6),
                "document_text": doc.page_content
            }

            semantic_results.append(row)

            similarity_results.append({
                "query_id": query_id,
                "query": query,
                "rank": rank,
                "document_id": metadata["document_id"],
                "title": metadata["title"],
                "similarity_score": round(float(score), 6)
            })

        # ----------------------------------
        # KEYWORD SEARCH USING TF-IDF
        # ----------------------------------

        query_matrix = tfidf.transform([query])

        keyword_scores = cosine_similarity(
            query_matrix,
            tfidf_matrix
        ).flatten()

        top_indices = np.argsort(keyword_scores)[::-1][:5]

        for rank, idx in enumerate(top_indices, 1):

            doc = DOCUMENTS[idx]

            keyword_results.append({
                "query_id": query_id,
                "query": query,
                "method": "Keyword Search",
                "rank": rank,
                "document_id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "source": f"dataset/documents/{doc['id']}.txt",
                "similarity_score": round(
                    float(keyword_scores[idx]), 6
                ),
                "document_text": doc["content"]
            })

    # ======================================
    # 7. SAVE CSV RESULTS
    # ======================================

    all_results = pd.DataFrame(
        semantic_results + keyword_results
    )

    all_results.to_csv(
        DATASET / "search_results.csv",
        index=False
    )

    pd.DataFrame(similarity_results).to_csv(
        DATASET / "similarity_scores.csv",
        index=False
    )

    print("\nSearch results saved successfully.")

    print("Semantic results:", len(semantic_results))
    print("Keyword results:", len(keyword_results))
    print("Similarity scores:", len(similarity_results))


# ==========================================
# 8. CREATE JUPYTER NOTEBOOK
# ==========================================

def create_notebook():

    notebook = nbformat.v4.new_notebook()

    notebook.cells = [

        nbformat.v4.new_markdown_cell(
            "# Week 12 - Semantic Document Search\n"
            "## LangChain and ChromaDB\n\n"
            "This notebook demonstrates semantic document retrieval, "
            "keyword search and similarity score comparison."
        ),

        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'dataset').exists():\n"
            "    ROOT = ROOT.parent\n\n"
            "results = pd.read_csv("
            "ROOT / 'dataset' / 'search_results.csv')\n"
            "scores = pd.read_csv("
            "ROOT / 'dataset' / 'similarity_scores.csv')\n\n"
            "print('Total search results:', len(results))\n"
            "print('Total similarity scores:', len(scores))"
        ),

        nbformat.v4.new_markdown_cell(
            "## 1. Display all documents and search results"
        ),

        nbformat.v4.new_code_cell(
            "display(results.head(10))"
        ),

        nbformat.v4.new_markdown_cell(
            "## 2. Compare Semantic Search and Keyword Search"
        ),

        nbformat.v4.new_code_cell(
            "query_id = 'query01'\n"
            "display(results[results['query_id'] == query_id]"
            ".sort_values(['method', 'rank']))"
        ),

        nbformat.v4.new_markdown_cell(
            "## 3. Similarity Scores"
        ),

        nbformat.v4.new_code_cell(
            "display(scores.head(15))"
        ),

        nbformat.v4.new_markdown_cell(
            "## 4. Search Method Comparison"
        ),

        nbformat.v4.new_code_cell(
            "comparison = results.groupby('method').agg(\n"
            "    total_results=('document_id', 'count'),\n"
            "    average_score=('similarity_score', 'mean')\n"
            ").reset_index()\n"
            "display(comparison)"
        )
    ]

    nbformat.write(
        notebook,
        NOTEBOOK_DIR / "Week12_Semantic_Document_Search.ipynb"
    )

    print("Jupyter notebook created.")


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

def main():

    print("=" * 50)
    print("WEEK 12 - SEMANTIC DOCUMENT SEARCH")
    print("=" * 50)

    DATASET.mkdir(parents=True, exist_ok=True)

    create_dataset()

    vector_store, embedding_model = generate_embeddings()

    perform_search(vector_store, embedding_model)

    create_notebook()

    print("\n" + "=" * 50)
    print("WEEK 12 PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()
