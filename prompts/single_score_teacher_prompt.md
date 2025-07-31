# Story Evaluation Task

You are given a short story consisting of a **context** and its **continuation**. Your task is to evaluate the quality of the continuation based on the following criteria:

## Evaluation Criteria

1. **Grammatical Correctness** (weight: 0.7)  
   Is the continuation free from grammatical or syntactic errors?

2. **Narrative Coherence** (weight: 0.2)  
   Does the continuation logically and smoothly follow the context?

3. **Creativity** (weight: 0.1)  
   Is the continuation imaginative or original?

Each criterion contributes to the final score according to the weight specified.

---

## Input

**Context:**  
`{{context}}`

**Continuation:**  
`{{continuation}}`

---

## Instructions

- Evaluate the continuation based on the three criteria above.
- Respond with a **single integer between `{{min_score}}` and `{{max_score}}`**, where:
  - `{{min_score}}` = Very poor quality  
  - `{{max_score}}` = Excellent quality
- Respond **only** with the score. Do **not** include any explanation or comments.

---