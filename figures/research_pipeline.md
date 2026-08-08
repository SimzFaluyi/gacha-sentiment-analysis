# Research Pipeline

The proposed computational workflow is:

┌──────────────────────────┐
│     Proposed Data Source │
│          Reddit          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      Data Validation     │
│ Missing values / dupes   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│       Preprocessing      │
│ Filtering / normalising  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Sentiment Labelling   │
│ Positive / Neutral / Neg │
└────────────┬─────────────┘
             │
             ▼
       ┌─────┴─────┐
       │           │
       ▼           ▼
   VADER       TF-IDF + LR
       │           │
       └─────┬─────┘
             │
             ▼
       ┌───────────┐
       │ DistilBERT│
       └─────┬─────┘
             │
             ▼
┌──────────────────────────┐
│        Evaluation        │
│ Precision / Recall / F1  │
│ Confusion Matrix         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Qualitative Analysis   │
│     Thematic Analysis    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      Interpretation      │
└──────────────────────────┘
