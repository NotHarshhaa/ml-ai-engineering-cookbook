# Module 14 — AI Infrastructure

The hardware and serving stack that makes AI fast and cheap. GPUs, mixed
precision, quantization, and vLLM-style paged inference.

| Notebook                                  | Topic                                                                  | Code walkthrough                                                            |
|-------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `01_gpu_computing.ipynb`                  | GPU thread-block model, PyTorch GPU API, mixed precision, FLOPS.   | [README_01_gpu_computing.md](README_01_gpu_computing.md)                     |
| `02_quantization.ipynb`                   | int8 / int4 / NF4, per-tensor vs per-channel, QLoRA-style loading. | [README_02_quantization.md](README_02_quantization.md)                       |
| `03_vllm_inference.ipynb`                 | PagedAttention, continuous batching, prefix caching, speculative. | [README_03_vllm_inference.md](README_03_vllm_inference.md)                   |
| `04_model_serving.ipynb`                  | http.server → FastAPI, Pydantic, SSE streaming, load testing.    | [README_04_model_serving.md](README_04_model_serving.md)                    |

Each notebook has a companion `README_<notebook-name>.md` that walks through
every code cell — open it alongside the notebook for a written explanation.
