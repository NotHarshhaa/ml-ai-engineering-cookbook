# Module 09 — Transformers & LLMs

The architecture that defines modern AI. Self-attention, multi-head attention,
BERT, GPT, and the scaling laws that connect them.

| Notebook                                  | Topic                                                                  | Code walkthrough                                                            |
|-------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `01_self_attention.ipynb`                | Scaled dot-product attention, causal mask, entropy.                  | [README_01_self_attention.md](README_01_self_attention.md)                    |
| `02_transformer_architecture.ipynb`       | Multi-head attention, positional encoding, transformer block, train. | [README_02_transformer_architecture.md](README_02_transformer_architecture.md) |
| `03_bert_mlm.ipynb`                       | MLM 80/10/10 masking, BERT inference, fine-tuning skeleton.          | [README_03_bert_mlm.md](README_03_bert_mlm.md)                                |
| `04_gpt_clm.ipynb`                        | Causal LM loss, top-k / top-p sampling, distilgpt2 generation.        | [README_04_gpt_clm.md](README_04_gpt_clm.md)                                  |

Each notebook has a companion `README_<notebook-name>.md` that walks through
every code cell — open it alongside the notebook for a written explanation of
what each block does and why.
