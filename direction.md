# ESG Indicator Thresholds (2018–2024)

A synthetic data

| Indicator | Unit | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Number of regenerated components | components | 3,000 | 5,000 | 6,500 | 8,000 | 10,000 | 12,000 | 14,000 |
| Weight of upcycled materials | tons | 1,200 | 1,800 | 2,200 | 2,500 | 3,200 | 4,500 | 5,800 |
| Number of repaired components | components | 6,000 | 8,000 | 10,000 | 12,000 | 14,000 | 18,000 | 22,000 |
| Reused parts | parts | 0.5M | 0.8M | 1.0M | 1.2M | 1.8M | 2.5M | 3.3M |
| Total purchased parts | parts | 12M | 11.8M | 11.5M | 11M | 11M | 11.5M | 12M |
| Recycled plastic used | tons | 200 | 400 | 600 | 800 | 1,000 | 1,200 | 1,600 |
| Total plastic used | tons | 7,500 | 7,200 | 6,800 | 6,000 | 5,800 | 5,000 | 4,500 |
| Energy from RES (self-use) | GWh | 5 | 12 | 18 | 25 | 32 | 45 | 60 |
| Total energy consumed | GWh | 420 | 415 | 410 | 400 | 390 | 380 | 370 |
| Power of RES plants | MW | 40 | 55 | 70 | 80 | 100 | 120 | 150 |
| Total plant power | MW | 200 | 220 | 240 | 250 | 280 | 310 | 330 |
| CO₂ captured | tons | 1,000 | 3,000 | 5,000 | 8,000 | 10,000 | 15,000 | 20,000 |
| CO₂ reused | % | 1% | 3% | 4% | 5% | 7% | 10% | 14% |
| Recovered water | m³ | 60k | 90k | 120k | 150k | 180k | 220k | 270k |
| Recyclable packaging | % of total | 60% | 65% | 72% | 80% | 85% | 95% | 98% |
| Total packaging | tons | 10,000 | 9,800 | 9,500 | 9,000 | 8,800 | 8,500 | 8,200 |
| Compostable packaging | tons | 50 | 150 | 300 | 500 | 800 | 1,200 | 1,600 |
| Reintegrated post-consumption mat. | tons | 100 | 180 | 250 | 300 | 450 | 650 | 900 |
| GHG emissions | Mt CO₂e | 2.8 | 2.7 | 2.6 | 2.5 | 2.3 | 2.1 | 1.9 |
| Baseline emissions (fixed 2020) | Mt CO₂e | 2.6 | 2.6 | 2.6 | 2.6 | 2.6 | 2.6 | 2.6 |
| Current emissions | Mt CO₂e | 2.8 | 2.7 | 2.6 | 2.3 | 2.2 | 1.9 | 1.8 |

# **Initial Experiments with Visual LLMs**

The goal:

- Test if a **Visual LLM** can extract ESG data from NFD pages.
- Compare it with **benchmarks**.
- Run it **page by page**.

---

### **Step 1: Define Prompt Template**

**Prompt template:**

```
In {benchmark_year}, the {industry} industry average for {indicator} was {benchmark_value}.
You are given a page from {company}'s sustainability report.
You must follow this direction:
 1. Look only at the information provided in this context. Do not use external knowledge.
 2. Identify {company}'s reported value for {indicator}. If no value is reported, answer "not found".
 3. Compare the company's value with the industry average:
    - If the company’s value is numerically higher than the industry average, answer "higher".
    - If the company’s value is numerically lower than the industry average, answer "lower".
    - If the company’s value equals the industry average, answer "equal".
 4. Format Your Response: Your entire response must strictly follow the format below. Do not add any introductory text or explanations.
    - output: {{higher | lower | equal | not found}},
    - extracted_value: {{extracted_value or not found}}

```



---

### **Step 2: Choose Models**

- **Qwen-3B-VL** or **Qwen-7B-VL** (supports images/PDF pages, efficient on 3060 GPU).
- **GPT-4o-mini** (via API, strong OCR + reasoning).
- **Yi-VL, LLaVA, MiniCPM-V** (lightweight alternatives).

 Note: test on a machine i9-12900H, 32GB RAM, RTX 3060 Mobile

---

### **Step 3: Execution Pipeline**

1. **Preprocess PDF** → split into pages (images or OCR text).
2. **For each page**:
    - Send prompt + page to model.
    - Collect response (`higher`, `lower`, `equal`, `not found`).
3. **Save results** into structured table e.g:

| Page | Indicator | Industry Benchmark | Model Output | Expected Result |
| --- | --- | --- |--------------|-----------------|
| 12 | CO₂ reused | 10% | higher       | lower           |
| 24 | Total plastic | 5000 tons | not found    | eqal            |

---

### **Step 4: Compare Models**

Run the same pipeline with:

- **Local model (e.g Qwen-3B-VL)** → check feasibility on your GPU.
- **API model (e.g gemini, gork, ...)** → check accuracy.

