# Quick Start: AI Learning System

## What Was Fixed

Your AI wasn't learning because the learning infrastructure existed but **wasn't connected to the web UI**. I've now integrated it.

## Changes Made

### 1. Enhanced Query Function (`web_app.py`)
- ✅ Now generates actual answers (not just retrieval)
- ✅ Injects learned knowledge from previous interactions
- ✅ Uses `ContextInjector` to retrieve relevant facts

### 2. Added Feedback UI
- ✅ 👍 Thumbs up button (rating: 0.9)
- ✅ 👎 Thumbs down button (rating: -0.5)
- ✅ ⭐ 1-5 star rating slider
- ✅ 💡 Correction text area for fixing wrong answers

### 3. Implemented Learning Pipeline
- ✅ `submit_feedback()` - Processes user ratings
- ✅ `FeedbackLearner` - Extracts facts from good answers
- ✅ Stores reasoning chains and query refinements
- ✅ Updates confidence scores based on usage

### 4. Added Knowledge Visibility
- ✅ "View Learned Facts" button
- ✅ "View Query Patterns" button
- ✅ Shows confidence, usage count, success rate

## How to Use

### Step 1: Start the App
```bash
streamlit run web_app.py
```

### Step 2: Ingest Documents
1. Go to "Knowledge Base" tab
2. Enter path to your documentation folder
3. Click "Ingest Documents"

### Step 3: Ask Questions
1. Type your question in the query box
2. Click "Search"
3. AI generates an answer using docs + learned knowledge

### Step 4: Provide Feedback
- Click 👍 if the answer is helpful
- Click 👎 if it's not helpful
- Or use the slider for custom rating
- Provide correction if the answer is wrong

### Step 5: See Learning Happen
- Expand "🧠 View Learned Knowledge"
- Click "View Learned Facts"
- See what the AI has extracted and learned!

## Example

**First Query:**
```
Q: "How do I set Spark executor memory?"
A: "Use spark.executor.memory=4g in your config..."
👍 (User clicks helpful)

→ AI learns: "spark.executor.memory controls executor heap size"
```

**Second Query (Later):**
```
Q: "What's the Spark memory config?"
A: [Uses learned fact from before]
   "Based on previous learning: spark.executor.memory 
   controls the heap size of executors..."

→ Answer is better because it remembers!
```

## Verification

Check if learning is working:

```sql
-- Connect to your PostgreSQL database
psql -h localhost -U omniscient -d omniscient

-- Check learned facts
SELECT topic, fact, confidence, usage_count 
FROM learned_facts 
ORDER BY confidence DESC;

-- Check feedback
SELECT question, rating, feedback_type, created_at 
FROM user_feedback 
ORDER BY created_at DESC 
LIMIT 10;
```

## What the AI Learns

1. **Facts** - Extracted from highly-rated answers (rating ≥ 0.6)
2. **Reasoning Chains** - Step-by-step logic that worked
3. **Query Refinements** - Better ways to phrase questions
4. **Corrections** - User-provided fixes for wrong answers

## Benefits

- ✅ **Continuous Improvement** - Gets smarter with every interaction
- ✅ **Domain Expertise** - Learns your project-specific knowledge
- ✅ **Error Correction** - Users can fix mistakes
- ✅ **Persistent** - Knowledge survives restarts
- ✅ **Transparent** - See what was learned

## Full Documentation

See `LEARNING_SYSTEM.md` for complete details on:
- Architecture and data flow
- Database schema
- Configuration options
- Advanced features
- Troubleshooting

## Summary

**The Problem:** Your AI had a sophisticated learning system built in, but it wasn't being used by the web app.

**The Solution:** I integrated the existing learning modules (`KnowledgeMemory`, `FeedbackLearner`, `ContextInjector`) into the web UI.

**The Result:** Your AI now learns from user feedback and improves over time! 🎉
