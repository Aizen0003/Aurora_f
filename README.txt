PROJECT AURORA - Self-Learning Notification Orchestrator
=======================================================

WHAT IT DOES
Aurora is a domain-agnostic, self-learning notification orchestrator. It figures
out WHO to talk to, WHAT to say, WHEN to say it, and continuously improves via
machine learning. Swap the PDF knowledge bank and user CSV - it auto-adapts to
any B2C/B2B domain (EdTech, FinTech, E-commerce, HealthTech, SaaS).

HOW IT WORKS
  Iteration 0 (Training):
    1. Knowledge Bank Engine ingests a company PDF via RAG-lite (PDF -> LLM ->
       TF-IDF cosine ranking) to extract North Star metric, feature-goal maps,
       tones, and Octalysis hooks.
    2. Data Ingestion auto-discovers any CSV schema via LLM, normalizes, and
       validates user behavioral data.
    3. Segmentation creates 6-12 MECE segments using RFM analysis + hierarchical
       clustering with optimal K selection (silhouette + Davies-Bouldin + elbow).
    4. ML Models train XGBoost churn prediction and LightGBM engagement forecasting.
    5. Goal Builder creates day-on-day journeys per segment x lifecycle stage.
    6. Theme Engine assigns Octalysis 8 Core Drives via LLM per segment.
    7. Template Generator creates exactly 5 bilingual (EN + Hinglish) templates
       per segment x lifecycle x goal x theme using LLM.
    8. Timing Optimizer uses survival analysis (Kaplan-Meier) across 6 time windows.
    9. Frequency Optimizer uses PS activeness thresholds with uninstall guardrails.
   10. Schedule Generator outputs per-user notification schedules showing journey
       progression.

  Iteration 1 (Learning):
    1. Performance Classifier labels templates GOOD/NEUTRAL/BAD per PS thresholds.
    2. Bayesian + Frequentist A/B testing with sequential analysis validates results.
    3. Multi-Armed Bandit (Thompson Sampling) updates template beliefs.
    4. BAD templates suppressed, GOOD templates promoted with higher weights.
    5. Timing and frequency re-optimized from experiment data.
    6. Delta Reporter documents every change with causal reasoning.
    7. Improved schedules generated showing measurable improvement.

QUICK START
  pip install -r requirements.txt
  set GROQ_API_KEY=your_key

  # Run both iterations (outputs auto-export to submission folders)
  python main.py --mode iteration0 --user-data user_data.csv --kb-pdf knowledge_bank.pdf
  python main.py --mode iteration1 --user-data user_data.csv --experiment-results experiment_results.csv

KEY ALGORITHMS
  - RFM + Ward Hierarchical Clustering (segmentation)
  - XGBoost (churn), LightGBM (engagement)
  - Thompson Sampling + UCB (template optimization)
  - Kaplan-Meier Survival Analysis (timing)
  - TF-IDF + Cosine Similarity (KB retrieval, NLP analysis)
  - Bayesian Beta-Binomial A/B Testing + Frequentist z-test

SUBMISSION STRUCTURE
  iteration_0_before_learning/
    company_north_star.json, feature_goal_map.json,
    allowed_tone_hook_matrix.json, user_segments.csv, segment_goals.csv,
    communication_themes.csv, message_templates.csv,
    timing_recommendations.csv, user_notification_schedule.csv
  iteration_1_after_learning/
    message_templates.csv (improved), timing_recommendations.csv (improved),
    user_notification_schedule.csv (improved), user_segments.csv
  experiment_results.csv (root)
  learning_delta_report.csv (root)

  Outputs are auto-exported to these folders after each iteration run.
