#!/usr/bin/env python

import os
import re
import glob
import argparse
import csv
import pickle

import torch
import numpy as np
import faiss

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model, PeftModel


# -------------------------------
# 1. PDF Parsing and Chunking for References
# -------------------------------
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfminer.six."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Splits text into overlapping chunks.
    """
    lines = re.split(r'\n+', text.strip())
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + " " + line
        else:
            current_chunk += " " + line
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def load_references(reference_folder):
    """
    Loads and processes all PDFs in a folder.
    Returns a list of dictionaries for each text chunk.
    """
    pdf_files = glob.glob(os.path.join(reference_folder, "*.pdf"))
    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing reference: {pdf_path} ...")
        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "doc": os.path.basename(pdf_path),
                "chunk_id": idx,
                "text": chunk
            })
    return all_chunks


# -------------------------------
# 2. Building and Saving/Loading the FAISS Index for References
# -------------------------------
def build_faiss_index(chunks, embed_model_name="gpt2"):
    """
    Creates embeddings for each chunk and builds a FAISS index.
    Returns the index, the embedding model, and the raw embeddings.
    """
    print("Loading embedding model for RAG...")
    embed_model = SentenceTransformer(embed_model_name)
    texts = [chunk["text"] for chunk in chunks]
    print("Computing embeddings for each chunk...")
    embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index, embed_model, embeddings


def save_rag(index, chunks, index_file, chunks_file):
    """Saves the FAISS index and chunks to disk."""
    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"RAG components saved to {index_file} and {chunks_file}")


def load_rag(index_file, chunks_file, embed_model_name):
    """Loads the FAISS index and chunks from disk."""
    print("Loading saved FAISS index and reference chunks...")
    index = faiss.read_index(index_file)
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)
    embed_model = SentenceTransformer(embed_model_name)
    return index, embed_model, chunks


def retrieve_relevant_chunks(query, index, embed_model, chunks, top_k=5):
    """
    Retrieves top_k chunks relevant to the query.
    """
    query_embed = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embed, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks


# -------------------------------
# 3. RAG Prompt Generation and LLM Inference
# -------------------------------
def generate_prompt(query, manuscript_content, reference_chunks=None):
    """
    Constructs a prompt that loads the entire manuscript into context,
    with optional supporting excerpts from references.
    """
    prompt = f"Research Query: {query}\n\n"
    prompt += "Manuscript Content:\n"
    prompt += manuscript_content + "\n\n"
    if reference_chunks:
        prompt += "Supporting Information from References:\n"
        for chunk in reference_chunks:
            prompt += f"[{chunk['doc']} - chunk {chunk['chunk_id']}]: {chunk['text']}\n\n"
    prompt += "Based on the above, please provide a detailed, research-focused answer primarily based on the manuscript."
    return prompt


def load_llm(base_model_name="gpt2"):
    """
    Loads a base LLM (default GPT-2) and creates a text-generation pipeline.
    """
    print(f"Loading base LLM: {base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator, model, tokenizer


# -------------------------------
# 4. Fine-Tuning with LoRA Using CSV Input
# -------------------------------
def load_finetune_csv(csv_file_path):
    """
    Loads fine-tuning examples from a CSV with no header.
    Columns: reference filename, query, answer.
    Returns a list of dictionaries with 'prompt' and 'completion' keys.
    """
    training_data = []
    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            reference = row[0].strip()
            query = row[1].strip()
            answer = row[2].strip()
            prompt = f"Reference: {reference}\nQuery: {query}\nAnswer:"
            completion = f" {answer}"
            training_data.append({"prompt": prompt, "completion": completion})
    return training_data


def fine_tune_model(base_model, tokenizer, training_data, output_dir):
    """
    Runs a fine-tuning loop using LoRA.
    training_data is a list of dictionaries with keys 'prompt' and 'completion'.
    Only the adapter weights are saved in the output_dir.
    """
    from torch.utils.data import Dataset, DataLoader

    class FineTuneDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=1024):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            full_text = item["prompt"] + item["completion"]
            encodings = self.tokenizer(
                full_text, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            input_ids = encodings.input_ids.squeeze()
            attention_mask = encodings.attention_mask.squeeze()
            return input_ids, attention_mask

    dataset = FineTuneDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"] if "gpt2" in tokenizer.name_or_path else ["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("Wrapping the base model with LoRA adapters...")
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.train()

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)
    num_epochs = 1  # Adjust as necessary
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_model.to(device)

    print("Starting fine tuning...")
    for epoch in range(num_epochs):
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
    print("Fine tuning complete. Saving adapter weights to", output_dir)
    peft_model.save_pretrained(output_dir)
    return peft_model


# -------------------------------
# 5. Main Pipeline
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM Research Companion Pipeline that loads the entire manuscript into context, "
                    "with optional supporting reference excerpts and CSV fine-tuning."
    )
    parser.add_argument("--reference_folder", type=str, help="Path to folder containing reference PDF files.")
    parser.add_argument("--manuscript_file", type=str, required=True,
                        help="Path to the in-progress manuscript PDF.")
    parser.add_argument("--query", type=str, default="",
                        help="A research query to test the system. If not provided, interactive mode is used.")
    parser.add_argument("--fine_tune", action="store_true",
                        help="Flag to run fine tuning on the base model using CSV training data.")
    parser.add_argument("--finetune_csv_file", type=str,
                        help="Path to CSV file with fine tuning examples (no header, columns: reference, query, answer).")
    parser.add_argument("--load_rag", action="store_true",
                        help="If set, load saved FAISS index and reference chunks instead of reprocessing references.")
    parser.add_argument("--save_rag", action="store_true",
                        help="If set, save the FAISS index and reference chunks after processing references.")
    parser.add_argument("--rag_index_file", type=str, default="rag_index.faiss",
                        help="File path for saving/loading the FAISS index for references.")
    parser.add_argument("--rag_chunks_file", type=str, default="rag_chunks.pkl",
                        help="File path for saving/loading the reference chunks.")
    parser.add_argument("--embed_model_name", type=str, default="gpt2",
                        help="Name of the embedding model to use for references.")
    parser.add_argument("--load_model_path", type=str,
                        help="Path to load a fine tuned adapter from.")
    parser.add_argument("--save_model_path", type=str, default="fine_tuned_model",
                        help="Path to save the fine tuned adapter.")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Name of the base LLM model to use.")
    args = parser.parse_args()

    # RAG Setup for References: Load or Build
    if args.load_rag:
        index, embed_model, chunks = load_rag(args.rag_index_file, args.rag_chunks_file, args.embed_model_name)
    else:
        if not args.reference_folder:
            print("Error: Reference folder must be provided if not loading a saved RAG index.")
            return
        print("=== Processing reference PDFs to build RAG ===")
        chunks = load_references(args.reference_folder)
        print(f"Extracted {len(chunks)} text chunks from references.")
        index, embed_model, _ = build_faiss_index(chunks, embed_model_name=args.embed_model_name)
        if args.save_rag:
            save_rag(index, chunks, args.rag_index_file, args.rag_chunks_file)

    # Load entire manuscript into context (not chunked or indexed)
    print("=== Loading manuscript PDF ===")
    manuscript_text = extract_text_from_pdf(args.manuscript_file)
    if not manuscript_text:
        print("Error: Could not extract text from the manuscript PDF. Exiting.")
        return
    manuscript_content = manuscript_text  # Full manuscript text as context

    # Model Setup: Load Fine-Tuned Adapter if Available, Otherwise Load Base Model
    if args.load_model_path:
        print(f"Loading fine tuned adapter from {args.load_model_path} ...")
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
        fine_tuned_model = PeftModel.from_pretrained(base_model, args.load_model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)
    else:
        print("=== Loading base LLM model ===")
        generator, base_model, tokenizer = load_llm(args.base_model)

    # Fine-Tuning Step
    if args.fine_tune:
        if not args.finetune_csv_file:
            print("Error: Fine tuning flag enabled but no CSV fine tuning file provided. Exiting.")
            return
        print("=== Loading fine tuning data from CSV ===")
        training_data = load_finetune_csv(args.finetune_csv_file)
        print(f"Loaded {len(training_data)} training examples from CSV.")
        print("=== Fine tuning the model with LoRA adapters ===")
        fine_tuned_model = fine_tune_model(base_model, tokenizer, training_data, output_dir=args.save_model_path)
        generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)

    # Retrieval-Augmented Generation (RAG)
    if args.query:
        query = args.query
        print(f"=== Processing query: {query} ===")
        # Optionally retrieve a supporting reference chunk
        reference_retrieved = retrieve_relevant_chunks(query, index, embed_model, chunks, top_k=1)
        prompt = generate_prompt(query, manuscript_content, reference_retrieved)
        print("\n=== Generated Prompt ===")
        print(prompt)
        print("\n=== Generating Answer ===")
        answer = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]["generated_text"]
        print("\n=== Answer ===")
        print(answer)
    else:
        print("=== Entering interactive mode. Type 'exit' to quit. ===")
        while True:
            query = input("Enter your research query: ")
            if query.lower() == "exit":
                break
            reference_retrieved = retrieve_relevant_chunks(query, index, embed_model, chunks, top_k=1)
            prompt = generate_prompt(query, manuscript_content, reference_retrieved)
            answer = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]["generated_text"]
            print("\n=== Answer ===")
            print(answer)
            print("====================================\n")


if __name__ == "__main__":
    main()
