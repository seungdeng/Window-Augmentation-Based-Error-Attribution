
# <p align="center">🌿 Window-Augmentation Based Methodology for Improving Error Attribution Performance in Multi-Agent Systems
</p>
<img width="1321" height="588" alt="image" src="https://github.com/user-attachments/assets/dbfa4712-b966-406e-aa7e-640f6d5584e2" />
<p align="center">
  <img width="539" height="864" alt="image" src="https://github.com/user-attachments/assets/b4e81470-badb-4157-a4cd-e09a1b05a1c8" />
</p>

---

## ✨ Overview

LLM-based multi-agent systems demonstrate strong performance across complex tasks.  
However, due to their **sequential interaction structure**, errors can propagate across steps, making it difficult to identify the true root cause.

This repository implements a **Window-Augmentation-based Error Attribution methodology**, which improves the identification of:

- 🎯 Decisive error step  
- 🤖 Responsible agent  

by leveraging **localized contextual information**.

---

## 🧩 Motivation

### ⚠️ Problem

Traditional failure attribution methods:

- Analyze logs **globally** or **step-by-step**
- Do not consider **interactions between adjacent steps**
- Fail in **cascading error scenarios**

---

### 💡 Solution

We propose:

> **Window Augmentation + Window-Focused Identification**

✔ Expand context around potential error steps  
✔ Capture inter-step interactions  
✔ Re-identify the *true* root cause  

---

## ⚙️ Method

### 🔄 Pipeline
<img width="820" height="749" alt="image" src="https://github.com/user-attachments/assets/fb4b125d-8970-4d2c-91d9-d6e188777a9c" />


---

### 🪟 Window Definition

For step \( s_i \):
<img width="337" height="26" alt="image" src="https://github.com/user-attachments/assets/ceb913db-aabf-46cd-9c82-e1a78be790bb" />


- `a`: window size  
- Dynamically adjusted at boundaries  

---

## 📊 Experimental Results

### 🧪 Dataset

- Algorithm Generated dataset (CaptainAgent-based logs)
- 126 samples  
- 5–10 interaction steps per sample  

---

### 📏 Metrics

- Step Accuracy  
- Agent Accuracy  

---

### 🏆 Main Results


| Method | Step Acc | Agent Acc |
|--------|---------|----------|
| (A) All-at-Once | 0.1746 | 0.4444 |
| (B) Step-by-Step | 0.2460 | 0.3571 |
| (C) + Window | 0.4444 | 0.5873 |
| (D) + Window (Sequential) | **0.4524** | **0.6270** |

✨ Improvements:
- Step Accuracy: **+83.9%**
- Agent Accuracy: **+75.6%**

---

### 📈 Window Size Analysis
<img width="1030" height="652" alt="image" src="https://github.com/user-attachments/assets/c6438e89-a517-4810-88b1-f417f02edf27" />


- Best performance at **window size = 3**
- Too small → insufficient context  
- Too large → noisy context  

---

## 🖥️ Environment

- Python 3.12  
- OpenAI GPT-4o  
- CPU: 24 cores  
- RAM: 32GB  

---

🌱 Contributions
Introduces window-based local context modeling
Improves failure attribution in multi-agent systems
Demonstrates effectiveness on real execution logs

⚠️ Limitations
Evaluated on a single dataset
Generalization not fully validated
Does not include error repair stage

---

MIT License

Copyright (c) 2026 SeungKeon Lee


