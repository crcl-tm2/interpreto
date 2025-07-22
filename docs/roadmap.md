# 📍 Interpreto Roadmap

Welcome to the roadmap of Interpreto 🪄. This document outlines the planned features and improvements for upcoming releases.

---

## 🧭 Upcoming Features

### 1. Evaluation Metrics for Attribution Methods

We plan to integrate a set of standardized evaluation metrics to assess the quality and reliability of attribution methods. We'll start by adding the **insertion/deletion metrics**.


### 2. Integration of VarGrad

We will implement **VarGrad**, a variance-based extension of gradient attribution. This method improves robustness by computing the **variance of attributions across multiple noisy forward passes**, capturing uncertainty and stabilizing explanations.

### 3. Integration of ConSim metric

We will implement **ConSim**, a robust evaluation metric measuring how well concept‑based explanations enable automated simulators (LLMs) to mimic the predictions of the original model. ConSim goes beyond evaluating just the concept space or importance alignment—it captures end-to-end interpretability effectiveness by testing whether the conveyed concepts actually allow a “simulator” to reproduce model outputs.

---

## 🙌 Contribute

Want to help shape the future of Interpreto? Check out our [contributing guide](contributing.md) and feel free to open an issue or pull request!
