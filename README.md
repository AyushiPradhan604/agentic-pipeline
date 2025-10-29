# 🧠 Agentic Research Pipeline
> Automatically convert research papers into structured poster summaries using multi-agent AI (powered by Qwen).

---

## 📘 Overview

This project implements a **three-agent pipeline** that reads a research paper (PDF or text), organizes its contents, summarizes each section, and prepares a **poster-ready structured format**.

### 🔹 Agents
1. **Section Identifier** → Extracts sections like *Introduction, Motivation, Dataset, Methodology, Results, Ablation, Conclusion*.
2. **Summarizer Agent** → Converts each section into concise **bullet points** while preserving image references.
3. **Poster Formatter** → Assembles all summarized sections into a final structured layout (poster-style format).

---

## 🧩 Folder Structure
