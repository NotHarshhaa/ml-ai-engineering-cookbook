# Module 15 — Fine-Tuning & Advanced AI

The parameter-efficient fine-tuning stack that made LLM customization
democratisable. PEFT, LoRA, QLoRA, SFT, and DPO — the standard modern AI
engineer's toolkit.

| Notebook                                  | Topic                                                                  | Code walkthrough                                                            |
|-------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `01_peft_lora.ipynb`                      | LoRA decomposition from scratch, merge, peft library, rank choice. | [README_01_peft_lora.md](README_01_peft_lora.md)                              |
| `02_qlora.ipynb`                          | NF4 quant, double quant, paged optimizers, 4-bit + LoRA.           | [README_02_qlora.md](README_02_qlora.md)                                    |
| `03_sft.ipynb`                            | Completion-only loss, chat templates, SFTTrainer.                  | [README_03_sft.md](README_03_sft.md)                                          |
| `04_dpo.ipynb`                            | DPO loss, preference data, beta, reward accuracy, IPO/KTO.        | [README_04_dpo.md](README_04_dpo.md)                                         |

Each notebook has a companion `README_<notebook-name>.md` that walks through
every code cell — open it alongside the notebook for a written explanation.
