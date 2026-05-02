# Lab 6 Fixes Summary

## Overview

Applied 4 specific fixes to Lab 6 (Makemore) to make it more beginner-friendly for students with neural network background but no NLP experience.

## Fixes Applied

### Fix 1: Updated Part 1 Tokenizer Example

**Location:** Part 1 - Character-Level Tokenizer

**Change:** Updated tokenizer example from "hello" to "friendly" showing three levels:
- Character level: `f, r, i, e, n, d, l, y` (8 tokens)
- Sub-word level (BPE): `friend, ly` (2 tokens)
- Word level: `friendly` (1 token)

**Reason:** Demonstrates the difference between tokenization levels more clearly.

**Files Modified:**
- `lab6_makemore.ipynb` (student)
- `lab6_makemore_solution.ipynb` (solution - already had this)

---

### Fix 2: More Specific Loss Function Explanation

**Location:** Part 3 - Loss Function - The Teacher

**Change:** Replaced generic loss explanation with makemore-specific details:
- Concrete example with name "anna" showing P('n') = 0.25
- Explained how model outputs 28 probabilities (one per character)
- Why negative log: examples with P=1.0, P=0.5, P=0.01
- Training examples: (<S>, 'a'), ('a', 'n'), ('n', 'n'), etc.
- Specific loss values: random ~3.3, trained ~2.4
- Perplexity interpretation: ~27 choices → ~11 choices

**Reason:** Original explanation was too abstract. Students need to see how loss applies specifically to bigram character prediction.

**Files Modified:**
- `lab6_makemore.ipynb` (student)
- `lab6_makemore_solution.ipynb` (solution)

---

### Fix 3: Fixed Bigram Visual Example

**Location:** Part 2 - Manual bigram example code

**Problem:** Code used string concatenation `'<S>' + name + '<E>'` which parsed `<S>` as three separate characters: `<`, `S`, `>`

**Fix:** Changed to use token lists:
```python
# OLD (broken):
full_name = '<S>' + name + '<E>'
for i in range(len(full_name) - 1):
    print(f"{full_name[i]}→{full_name[i+1]}")

# NEW (correct):
tokens = ['<S>'] + list(name) + ['<E>']
for i in range(len(tokens) - 1):
    print(f"{tokens[i]}→{tokens[i+1]}")
```

**Result:**
- OLD output: `<→S, S→>, >→s, s→a, ...` (wrong!)
- NEW output: `<S>→s, s→a, a→m, m→<E>` (correct!)

**Files Modified:**
- `lab6_makemore.ipynb` (student)
- `lab6_makemore_solution.ipynb` (solution)

---

### Fix 4: Moved Exercise 2 After Model Creation

**Location:** Moved Exercise 2 from Part 3 to Part 5 (after Exercise 3)

**New Order:**
1. Part 3: Loss Function - The Teacher (explanation only)
2. Part 4: Building the Bigram Dataset
3. Part 5: Training a Bigram Model
   - Exercise 3: Implement Bigram Model
   - **Exercise 2: Calculate Loss on Random Model** (moved here)
   - Exercise 4: Implement Training Loop

**New Exercise 2 Content:**
- Title: "Calculate Loss on Random Model"
- Context: "Now that we have a model (randomly initialized)..."
- Task: Calculate loss on untrained model to see baseline (~3.3)
- Shows where we START before training

**Reason:** Students need a model instance first before they can calculate loss on it. Original order had Exercise 2 before model creation, which was confusing.

**Files Modified:**
- `lab6_makemore.ipynb` (student - with TODOs)
- `lab6_makemore_solution.ipynb` (solution - complete implementation)

---

## Validation

All notebooks validated for:
- ✓ GitHub compatibility (correct JSON structure)
- ✓ Markdown cells: no `outputs` or `execution_count` fields
- ✓ Code cells: have `outputs` (empty array) and `execution_count` (null)
- ✓ Solution notebook: all exercises have working implementations
- ✓ Student notebook: all exercises have TODOs and hints

## Testing

Verified bigram example output:
```
Bigram pairs (current → next):
  sam: <S>→s, s→a, a→m, m→<E>, 
  max: <S>→m, m→a, a→x, x→<E>, 
  alex: <S>→a, a→l, l→e, e→x, x→<E>, 

Bigram count table:
  After '<S>': a:1  m:1  s:1  
  After 'a': l:1  m:1  x:1  
  After 'x': <E>:2
```

Special tokens now correctly treated as single tokens!

---

## Impact

These fixes make Lab 6 significantly more beginner-friendly:
1. Clearer tokenization concepts with concrete example
2. Loss function grounded in specific bigram prediction task
3. Correct special token handling (no confusing character-by-character parsing)
4. Logical exercise flow (create model → test random loss → train → evaluate)

Students can now follow the progression from concepts → implementation → training without confusion about when model exists or why special tokens are being parsed wrong.
