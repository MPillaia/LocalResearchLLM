# LocalResearchLLM

  1. Clone the [LocalResearchLLM repository](https://github.com/MPillaia/LocalResearchLLM)  
     &nbsp;&nbsp;&nbsp;&nbsp;- Note that sample manuscript and references are not provided to prevent improper distribution of published and unpublished work.
  2. Install required dependencies with `pip install -r requirements.txt`.
  3. Create a directory named "references" and include PDF files of your desired references.
  4. Create your fine-tuning CSV file. Any set of question/answer pairs are acceptable, as long as the CSV is in the following format:

     | Reference file name | Question         | Answer         |
     |---------------------|------------------|----------------|
     | reference1.pdf      | Sample Question  | Sample Answer  |

  5. Create a PDF copy of your manuscript.
  6. To complete the RAG processing and fine-tune the base model, run:
     ```bash
     python assistant.py --manuscript_file /path/to/manuscript/my_manuscript.pdf --reference_folder /path/to/references --save_rag --fine_tune --finetune_csv_file /path/to/data/finetuning_samples.csv 
     ```
  7. After completion, to run your research companion without re-processing, run:
     ```bash
     python assistant.py --manuscript_file /path/to/manuscript/my_manuscript.pdf --load_rag --load_model_path /path/to/fine_tuned_model
     ```
     &nbsp;&nbsp;&nbsp;&nbsp;- Note that by default the RAG files and `fine_tuned_model` directory will be stored in your working directory. You can specify the specific RAG file locations with the `--rag_index_file` and `--rag_chunks_file` flags.

---
