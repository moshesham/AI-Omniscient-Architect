# 🧠 AI Learning System - Implementation Guide

## Overview

Your AI system now has **persistent learning capabilities** that allow it to improve over time based on user feedback. The learning infrastructure was already built into your codebase but wasn't being used - I've now integrated it into the web app.

## What Changed

### ✅ Learning Features Added to Web App

1. **Answer Generation with Learned Context**
   - AI now generates actual answers (not just retrieval)
   - Injects learned knowledge from previous interactions
   - Uses both retrieved documents + learned facts

2. **User Feedback Collection**
   - 👍 Thumbs up/down buttons
   - ⭐ Custom 1-5 star ratings
   - 💡 Correction submission for wrong answers

3. **Automatic Learning from Feedback**
   - Extracts facts from highly-rated answers
   - Stores reasoning chains that worked
   - Learns query refinement patterns
   - Updates confidence scores based on usage

4. **Knowledge Persistence**
   - All learning stored in PostgreSQL database
   - Survives restarts and sessions
   - Improves over time automatically

5. **Visibility into Learning**
   - View learned facts with confidence scores
   - See query refinement patterns
   - Track what knowledge is being used

## How It Works

### Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  1. Retrieve Learned Knowledge      │
│     (Facts, Reasoning, Refinements) │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  2. Search Knowledge Base           │
│     (Hybrid: Vector + BM25)         │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  3. Generate Answer                 │
│     (LLM: Context + Learned Facts)  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  4. Present Answer + Sources        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  5. Collect User Feedback           │
│     (Rating, Correction)            │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  6. Learn from Feedback             │
│     • Extract facts                 │
│     • Store reasoning               │
│     • Learn refinements             │
│     • Update confidence             │
└─────────────────────────────────────┘
```

### Database Schema

The learning system uses 4 PostgreSQL tables:

**1. `learned_facts`** - Extracted knowledge
```sql
- topic: Category/subject
- fact: The learned fact
- source_question: Original question
- source_answer: Original answer
- confidence: 0-1 score
- usage_count: Times used
- success_rate: How often it helped
- embedding: Vector for similarity search
```

**2. `reasoning_chains`** - Successful reasoning patterns
```sql
- question: Original question
- steps: JSONB array of reasoning steps
- final_answer: The answer
- was_correct: User validated it
- feedback_score: User rating
- embedding: Vector for similarity
```

**3. `query_refinements`** - Query improvement patterns
```sql
- original_query: User's original query
- refined_query: Better version
- improvement_score: How much it helped
- times_applied: Usage count
- success_rate: Success percentage
```

**4. `user_feedback`** - Raw feedback log
```sql
- question: User question
- answer: AI answer
- feedback_type: POSITIVE/NEGATIVE/CORRECTION
- correction: User's correct answer
- rating: -1 to 1
- context_chunks: Which chunks were used
```

## Using the Learning System

### Step 1: Ingest Documents
```python
# In web app: Knowledge Base tab
1. Enter document folder path
2. Select chunking strategy (auto recommended)
3. Click "Ingest Documents"
```

### Step 2: Ask Questions
```python
# In web app: Query section
1. Type your question
2. Click "Search"
3. AI generates answer using:
   - Retrieved document chunks
   - Previously learned facts
   - Reasoning patterns
```

### Step 3: Provide Feedback
```python
# After receiving an answer:
1. Click 👍 if helpful (rating: 0.9)
2. Click 👎 if not helpful (rating: -0.5)
3. Or use slider for custom 1-5 stars
4. Optionally provide correction
```

### Step 4: AI Learns Automatically
When you provide feedback, the system:
- **Positive feedback** (rating ≥ 0.6):
  - Extracts facts from the answer
  - Stores reasoning chain if step-by-step
  - Learns query refinement if query was improved
  
- **Negative feedback** (rating < 0):
  - If you provide correction: stores as high-confidence fact
  - Updates confidence scores downward

### Step 5: View Learned Knowledge
```python
# In Knowledge Base tab, expand "View Learned Knowledge"
- Click "View Learned Facts" to see extracted facts
- Click "View Query Patterns" to see refinements
- See confidence, usage count, success rate
```

## Example Learning Cycle

### First Interaction
```
User: "How do I configure Spark executor memory?"

AI: [Searches docs, no learned knowledge yet]
"Use spark.executor.memory configuration property..."

User: 👍 (Helpful)

AI Learns:
✅ Extracted fact: "spark.executor.memory controls executor heap size"
✅ Confidence: 0.65
```

### Second Interaction (Same Topic)
```
User: "What's the Spark executor memory setting?"

AI: [Injects learned fact from interaction 1]
"Based on previous learning: spark.executor.memory controls 
the heap size of each executor JVM. You can set it like..."

[Answer is better because it uses learned knowledge]

User: ⭐⭐⭐⭐⭐ (5 stars)

AI Learns:
✅ Updated fact confidence: 0.65 → 0.75
✅ Increased usage count: 1 → 2
✅ Success rate: 100%
```

### Correction Example
```
User: "What port does PostgreSQL use?"

AI: "PostgreSQL uses port 3306"

User: 💡 Correction: "PostgreSQL uses port 5432, not 3306"

AI Learns:
✅ Stored correction as high-confidence fact (0.8)
✅ Topic: "PostgreSQL"
✅ Fact: "PostgreSQL uses port 5432"
```

## Configuration

### Learning Behavior (in code)
```python
from omniscient_rag.learning import LearningConfig

config = LearningConfig(
    min_positive_rating=0.6,          # Min rating to learn from
    auto_extract_facts=True,           # Extract facts from answers
    auto_detect_reasoning=True,        # Detect step-by-step reasoning
    learn_query_patterns=True,         # Learn query refinements
    min_retrieval_improvement=0.2,     # Min improvement to store
    positive_confidence_boost=0.1,     # Confidence increase
    negative_confidence_penalty=0.15,  # Confidence decrease
)
```

### Context Injection (in code)
```python
from omniscient_rag.learning import InjectionConfig

config = InjectionConfig(
    max_facts=5,                       # Top N facts to inject
    max_reasoning_chains=2,            # Top N reasoning patterns
    max_corrections=3,                 # Top N corrections
    include_few_shot_reasoning=True,   # Include examples
    min_fact_confidence=0.3,           # Confidence threshold
    max_context_tokens=2000,           # Max context size
)
```

## Benefits

### 1. **Continuous Improvement**
- AI gets smarter with every interaction
- No manual retraining needed
- Domain expertise accumulates

### 2. **Domain Specialization**
- Learns project-specific terminology
- Remembers common patterns
- Adapts to your use case

### 3. **Error Correction**
- Users can correct mistakes
- Corrections stored as high-confidence facts
- Won't repeat the same error

### 4. **Query Understanding**
- Learns how users ask questions
- Stores successful query reformulations
- Improves retrieval over time

### 5. **Transparency**
- View what was learned
- See confidence scores
- Track usage and success

## Advanced Features

### Contrastive Learning (Future)
```python
from omniscient_rag.learning import ContrastiveLearningHelper

helper = ContrastiveLearningHelper(memory)
pairs = await helper.generate_training_pairs(min_samples=100)
# Export for fine-tuning embeddings
jsonl = helper.export_for_fine_tuning(pairs, output_format="jsonl")
```

### Batch Learning from History
```python
# Learn from historical interactions
feedback_history = [
    {
        "question": "...",
        "answer": "...",
        "rating": 0.9,
        "feedback_type": "positive",
    },
    # ... more
]

results = await learner.batch_learn_from_history(feedback_history)
# Results: {processed: 50, facts_extracted: 12, chains_stored: 5}
```

### Knowledge Evaluation
```python
# Measure knowledge quality
from omniscient_rag.metrics import KnowledgeScorer

scorer = KnowledgeScorer(store, searcher, llm_fn)
score = await scorer.evaluate()
# Score: {overall: 85, coverage: 80, accuracy: 90, ...}
```

## Troubleshooting

### Learning Not Working?
1. **Check database connection**
   ```bash
   # Ensure PostgreSQL is running
   docker-compose ps postgres
   ```

2. **Verify tables exist**
   ```sql
   SELECT table_name FROM information_schema.tables 
   WHERE table_schema = 'public' 
   AND table_name LIKE 'learned_%';
   ```

3. **Check learning module import**
   ```python
   # In web_app.py, check:
   print(HAS_LEARNING)  # Should be True
   ```

### No Learned Knowledge Showing?
1. **Provide feedback first** - Nothing learned until users rate answers
2. **Check rating threshold** - Default minimum is 0.6 (3/5 stars)
3. **View database directly**:
   ```sql
   SELECT COUNT(*) FROM learned_facts;
   SELECT topic, fact, confidence FROM learned_facts ORDER BY confidence DESC;
   ```

### AI Not Using Learned Knowledge?
1. **Check context injection** - Should see "🧠 Using Learned Knowledge" in UI
2. **Verify confidence threshold** - Default minimum is 0.3
3. **Check query relevance** - Learned facts must match query semantically

## Demo Script

Want to see it in action? Run this:

```bash
cd scripts
python demo_knowledge_persistence.py
```

This demonstrates:
- Storing learned facts
- Injecting knowledge into new sessions
- Feedback processing
- Query refinement learning

## Next Steps

### Recommended Improvements

1. **Add More Feedback Types**
   - "Missing information" button
   - "Too verbose" / "Too brief"
   - Relevance rating

2. **Implement Confidence Decay**
   - Old unused facts lose confidence over time
   - Forces re-validation

3. **Active Learning**
   - AI asks clarifying questions
   - Requests feedback on uncertain answers

4. **Multi-User Learning**
   - Aggregate feedback across users
   - Consensus-based confidence

5. **Learning Analytics**
   - Dashboard showing learning trends
   - Most valuable facts
   - Topic coverage gaps

## Summary

**Before**: Your AI did basic retrieval but never learned or improved.

**After**: Your AI:
- ✅ Learns facts from highly-rated answers
- ✅ Remembers successful reasoning patterns
- ✅ Improves query understanding over time
- ✅ Persists knowledge across sessions
- ✅ Gets better with every user interaction
- ✅ Shows transparency in what it learned

The learning system is **fully integrated** and **ready to use**. Just start asking questions and providing feedback!

---

**Questions?** See the demo script or check the RAG module documentation:
- `packages/rag/src/omniscient_rag/learning/`
- `scripts/demo_knowledge_persistence.py`
