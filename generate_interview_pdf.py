import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages after cover/first page)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "PROJECT AURORA | Technical Architecture & ML Interview Playbook")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
        
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 32, footer_text)
        self.drawString(54, 32, "CONFIDENTIAL - PREPARED FOR SYSTEM ARCHITECTURE & ML INTERVIEW")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 44, 8.5 * 72 - 54, 44)
        
        self.restoreState()

def build_pdf(filename="Project_Aurora_Technical_Architecture_and_Interview_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1A365D")   # Deep navy
    c_secondary = colors.HexColor("#2B6CB0") # Slate blue
    c_accent = colors.HexColor("#D69E2E")    # Warm gold
    c_dark = colors.HexColor("#2D3748")      # Charcoal body
    c_light_bg = colors.HexColor("#F7FAFC")  # Off-white / light slate
    c_border = colors.HexColor("#E2E8F0")
    c_box_border = colors.HexColor("#CBD5E0")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=c_primary,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )
    
    formula_style = ParagraphStyle(
        'Formula',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2C5282"),
        spaceAfter=4,
        spaceBefore=4
    )
    
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_dark
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=c_primary
    )
    
    table_cell_header = ParagraphStyle(
        'TableHeader',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )

    story = []

    # Title Banner Block
    story.append(Paragraph("PROJECT AURORA", title_style))
    story.append(Paragraph("Domain-Generic Self-Learning Notification System & ML Engine", subtitle_style))
    story.append(Paragraph("<b>Comprehensive Technical Architecture, Mathematical Foundations & Interview Playbook</b>", ParagraphStyle('SubSub', parent=body_style, fontSize=9.5, leading=14, textColor=c_primary)))
    story.append(HRFlowable(width="100%", thickness=2, color=c_primary, spaceBefore=4, spaceAfter=12))

    # Executive Summary Box
    summary_text = (
        "<b>Executive Summary:</b> Project Aurora is an automated, closed-loop notification orchestration engine designed to be "
        "completely domain-generic. Operating across two discrete iterations (<b>Iteration 0: Cold Start</b> and <b>Iteration 1: Closed-Loop Learning</b>), "
        "Aurora ingests unstructured organizational knowledge (handbooks/PDFs) alongside user behavioral event logs. It builds dynamic schema mappings, "
        "segments users via RFM and Agglomerative Hierarchical clustering, predicts churn and engagement intensity via gradient boosted decision trees (XGBoost/LightGBM), "
        "schedules deliveries using Kaplan-Meier survival hazard modeling, and continuously optimizes copy performance through Beta-Bernoulli Thompson Sampling bandits "
        "and NLP token attribution."
    )
    summary_table = Table([[Paragraph(summary_text, callout_style)]], colWidths=[504])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF8FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BEE3F8")),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # Architecture Overview Table
    arch_data = [
        [Paragraph("Pipeline Phase", table_cell_header), Paragraph("Key Algorithms & Modules", table_cell_header), Paragraph("Primary Technical Role", table_cell_header)],
        [Paragraph("1. Knowledge Bank", table_cell_bold), Paragraph("Groq LLM (Llama 3.3 70B), JSON Schema", table_cell), Paragraph("Zero-shot extraction of North Star, feature maps, tone matrices.", table_cell)],
        [Paragraph("2. Ingestion & Schema", table_cell_bold), Paragraph("Cosine Similarity, Robust Min-Max", table_cell), Paragraph("Domain-agnostic column role detection and propensity engineering.", table_cell)],
        [Paragraph("3. Segmentation", table_cell_bold), Paragraph("RFM Quintiles, Ward Hierarchical Linkage", table_cell), Paragraph("Determines MECE cohorts (6-12) bounded by silhouette score.", table_cell)],
        [Paragraph("4. Propensity ML", table_cell_bold), Paragraph("XGBoost Classifier & LightGBM Regressor", table_cell), Paragraph("Predicts binary churn risk and continuous engagement intensity.", table_cell)],
        [Paragraph("5. Timing & Survival", table_cell_bold), Paragraph("Kaplan-Meier Estimator, Hazard Rates", table_cell), Paragraph("Identifies peak drop-off windows to schedule preemptive nudges.", table_cell)],
        [Paragraph("6. Template Generation", table_cell_bold), Paragraph("Combinatorial Matrix (EN + HI)", table_cell), Paragraph("Synthesizes bilingual copy crossing goals, hooks, and segments.", table_cell)],
        [Paragraph("7. Multi-Armed Bandit", table_cell_bold), Paragraph("Thompson Sampling (Beta-Bernoulli)", table_cell), Paragraph("Dynamic Bayesian exploration-exploitation over template arms.", table_cell)],
        [Paragraph("8. Significance & Stats", table_cell_bold), Paragraph("Two-sample t-test, Bonferroni, Cohen's h", table_cell), Paragraph("Validates statistical confidence (>95%) before copy promotion.", table_cell)],
        [Paragraph("9. NLP Optimization", table_cell_bold), Paragraph("TF-IDF Vectorization, Token Attribution", table_cell), Paragraph("Attaches engagement/fatigue signals to specific phrases & hooks.", table_cell)],
    ]
    t_arch = Table(arch_data, colWidths=[110, 194, 200])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 14))

    # SECTION 1: Knowledge Bank Engine
    story.append(Paragraph("1. Knowledge Bank Engine (Domain-Generic RAG-Lite)", h1_style))
    story.append(Paragraph(
        "<b>Implementation (<code>src/knowledge_bank/kb_engine.py</code>):</b> Project Aurora uses an in-memory RAG-lite architecture. "
        "It extracts unstructured documentation using PDF parsers or raw text ingestion, feeds the text into Groq API LLM endpoints "
        "(<code>llama-3.3-70b-versatile</code> or <code>mixtral-8x7b-32768</code>), and applies constrained JSON schema decoding at low temperature (T=0.1) "
        "to extract: (1) Company North Star metric, (2) Feature-to-Goal maps, and (3) Tone/Hook boundary constraints.",
        body_style
    ))
    story.append(Paragraph("Mathematical & Algorithmic Formulation:", h2_style))
    story.append(Paragraph("Cosine similarity ranking for semantic feature matching:", body_style))
    story.append(Paragraph("cos(u, v) = (u · v) / (||u||₂ ||v||₂) = ∑(uᵢ vᵢ) / ( √(∑ uᵢ²) √(∑ vᵢ²) )", formula_style))
    story.append(Paragraph("TF-IDF weighting formulation across vocabulary t in document d:", body_style))
    story.append(Paragraph("TF-IDF(t, d, D) = TF(t, d) × log( (1 + |D|) / (1 + |{d ∈ D : t ∈ d}|) ) + 1", formula_style))
    story.append(Paragraph(
        "<b>Architectural Tradeoff:</b> An in-memory prompt-driven extraction was selected over an external vector database (Chroma, Pinecone). "
        "For single handbook documents (&lt;50 pages), deploying and managing vector embeddings and approximate nearest neighbor (ANN) indexes "
        "introduces needless infrastructure complexity without measurable recall gain.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Production Alternatives:</b> Hybrid retrieval combining BM25 (lexical) with Dense Bi-encoders (SentenceTransformers <code>all-MiniLM-L6-v2</code>) "
        "via Reciprocal Rank Fusion: <i>RRF(d) = ∑ 1 / (60 + rank(d))</i>, combined with DSPy for automated prompt parameter compilation.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # SECTION 2: Dynamic Schema Ingestion
    story.append(Paragraph("2. Dynamic Schema Mapping & Feature Ingestion", h1_style))
    story.append(Paragraph(
        "<b>Implementation (<code>src/intelligence/data_ingestion.py</code>):</b> To achieve domain-agnostic execution, raw tabular data cannot be "
        "assumed to follow fixed column names (e.g., 'user_id', 'sessions'). Ingestion computes semantic similarity between dataset headers and canonical "
        "behavioral prototypes, automatically mapping columns to: Identifiers, Recency timestamps, Frequency counters, Monetary spend, and Product usage metrics.",
        body_style
    ))
    story.append(Paragraph("Feature Normalization & Robust Scaling:", h2_style))
    story.append(Paragraph("Robust Median-IQR Normalization prevents extreme user outliers from distorting cluster centroids:", body_style))
    story.append(Paragraph("x_scaled = (x - median(x)) / (IQR(x) + ε),  where IQR = Q₇₅ - Q₂₅", formula_style))
    story.append(Paragraph(
        "<b>Engineered Propensities:</b> Computes composite scores including <i>Activeness Score</i> (weighted recency decay × session volume) "
        "and sub-feature affinities (Gamification, Social, Productivity, AI Tutor) normalized to [0, 1].",
        body_style
    ))
    story.append(Spacer(1, 10))

    # SECTION 3: Segmentation
    story.append(Paragraph("3. Customer Segmentation (RFM + Hierarchical Clustering)", h1_style))
    story.append(Paragraph(
        "<b>Implementation (<code>src/intelligence/segmentation.py</code>):</b> Computes RFM quintile scores (ranks 1 to 5) for Recency, Frequency, "
        "and Monetary dimensions. It then conducts a sweep across K ∈ [6, 12] clusters using Agglomerative Hierarchical Clustering with Ward's linkage, "
        "selecting the partition that maximizes the global Silhouette Coefficient while enforcing a minimum segment size threshold of 5%.",
        body_style
    ))
    story.append(Paragraph("Ward's Linkage Variance Criterion:", h2_style))
    story.append(Paragraph("At each step, merges clusters A and B that minimize the increase in total within-cluster sum of squares:", body_style))
    story.append(Paragraph("ΔESS_AB = ( (n_A · n_B) / (n_A + n_B) ) × || m_A - m_B ||₂²", formula_style))
    story.append(Paragraph("Silhouette Score Validation:", h2_style))
    story.append(Paragraph("s(i) = ( b(i) - a(i) ) / max( a(i), b(i) ),  s(i) ∈ [-1, 1]", formula_style))
    story.append(Paragraph("where a(i) is mean intra-cluster distance and b(i) is smallest mean distance to any other cluster.", body_style))
    story.append(Paragraph(
        "<b>Design Tradeoffs:</b> K-Means is computationally fast (O(N·K·I)), but non-deterministic and assumes isotropic, spherical clusters. "
        "Ward's Hierarchical clustering requires O(N²) memory for the distance matrix, but produces strictly deterministic, MECE hierarchical groupings. "
        "For massive scale (N > 1M), a production alternative is <b>BIRCH</b> (CF-trees) or <b>Mini-Batch K-Means</b> combined with UMAP projection.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # SECTION 4: Propensity Modeling
    story.append(Paragraph("4. Supervised Propensity Modeling (XGBoost & LightGBM)", h1_style))
    story.append(Paragraph(
        "<b>Implementation (<code>src/intelligence/ml_propensity_models.py</code>):</b> Implements a dual-model predictive architecture: "
        "(1) <b>XGBoost Classifier</b> predicting user churn risk using <code>scale_pos_weight</code> to handle class imbalance, and "
        "(2) <b>LightGBM Regressor</b> predicting future 7-day engagement intensity. Evaluated using 5-fold Stratified Cross-Validation.",
        body_style
    ))
    story.append(Paragraph("Gradient Boosted Trees Optimization Objective:", h2_style))
    story.append(Paragraph("Second-order Taylor expansion of the loss function at boosting round m:", body_style))
    story.append(Paragraph("L^(m) ≈ ∑ [ gᵢ f_m(xᵢ) + 0.5 hᵢ f_m²(xᵢ) ] + γ T + 0.5 λ ∑ wⱼ²", formula_style))
    story.append(Paragraph("where gᵢ = ∂_ŷ l(yᵢ, ŷ^(m-1)) (gradient) and hᵢ = ∂²_ŷ l(yᵢ, ŷ^(m-1)) (Hessian).", body_style))
    story.append(Paragraph("Class Imbalance Compensation:", h2_style))
    story.append(Paragraph("scale_pos_weight = N_negative / N_positive", formula_style))
    story.append(Paragraph(
        "<b>The AUC 0.44 Diagnostic (Crucial Interview Topic):</b> On the synthetic sample data, the churn classifier produced an AUC of ~0.44. "
        "An AUC below 0.5 indicates performance worse than a coin flip (anti-predictive noise or label disconnect). "
        "In Project Aurora, circular target leakage was intentionally removed (prior iterations leaked target-derived metrics, artificially yielding 1.0). "
        "In production, resolving this requires an explicit temporal observation/prediction split (e.g., 30-day feature window, 14-day outcome window) "
        "and PR-AUC / Focal Loss optimization.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # SECTION 5: Timing & Survival Analysis
    story.append(Paragraph("5. Send-Time Optimization via Survival Analysis", h1_style))
    story.append(Paragraph(
        "<b>Implementation (<code>src/communication/timing_optimizer.py</code>):</b> Replaces static broadcast scheduling with survival modeling. "
        "Models user drop-off time using the <b>Kaplan-Meier Estimator</b> to identify hazard peaks—the exact elapsed hours when user engagement "
        "is most vulnerable to cessation—triggering re-engagement nudges prior to complete attrition.",
        body_style
    ))
    story.append(Paragraph("Kaplan-Meier Product-Limit Estimator:", h2_style))
    story.append(Paragraph("Ŝ(t) = ∏_{tᵢ ≤ t} ( 1 - dᵢ / nᵢ )", formula_style))
    story.append(Paragraph("where dᵢ is the count of churn/drop-off events at time tᵢ, and nᵢ is the number of users at risk right before tᵢ.", body_style))
    story.append(Paragraph("Instantaneous Hazard Rate Modeling:", h2_style))
    story.append(Paragraph("λ(t) = lim_{Δt→0} P( t ≤ T < t + Δt | T ≥ t ) / Δt", formula_style))
    story.append(Paragraph(
        "<b>Why Survival Analysis over Standard Regression:</b> Standard regression cannot handle <b>right-censored data</b> "
        "(users who have not yet churned at observation cutoff). Imputing arbitrary values creates systemic bias. Survival models properly estimate "
        "survival curves under heavy right-censorship. Production upgrade: <b>Cox Proportional Hazards Model</b> conditioning on individual feature vectors x.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # SECTION 6: Template Generation
    story.append(Paragraph("6. Domain-Generic Template Engine & Message Synthesis", h1_style))
    story.append(Paragraph(
        "<b>Implementation (<code>src/communication/template_generator.py</code>):</b> Automatically synthesizes 600+ bilingual notification templates "
        "(English and Hindi/Hinglish). The generator spans a full combinatorial matrix crossing: 6 User Segments × 6 Communication Themes "
        "(Streaks, Milestones, Social Proof, Dormant Re-engagement, Feature Discovery, Gamification) × 4 Psychological Hooks (Loss Aversion, Social Urgency, "
        "Curiosity, Encouragement). Templates are assigned deterministic IDs (<code>TPL_0001</code> to <code>TPL_0600</code>).",
        body_style
    ))
    story.append(Spacer(1, 10))

    # SECTION 7: Multi-Armed Bandit
    story.append(Paragraph("7. Multi-Armed Bandit Engine (Thompson Sampling)", h1_style))
    story.append(Paragraph(
        "<b>Implementation (<code>src/learning/multi_armed_bandit.py</code>):</b> Closed-loop learning across iterations is governed by Bayesian "
        "Multi-Armed Bandits. Rather than running static 50/50 A/B tests that incur massive regret, Aurora implements <b>Thompson Sampling</b> "
        "using conjugate <b>Beta-Bernoulli</b> distributions to balance exploration of new templates against exploitation of proven high-CTR winners.",
        body_style
    ))
    story.append(Paragraph("Conjugate Beta Posterior Formulation:", h2_style))
    story.append(Paragraph("Prior: θ_k ~ Beta(α₀, β₀). Observed data: s_k successes (clicks), f_k failures (ignores).", body_style))
    story.append(Paragraph("Posterior: P(θ_k | s_k, f_k) = Beta( α₀ + s_k,  β₀ + f_k )", formula_style))
    story.append(Paragraph("Posterior Mean & Variance:", body_style))
    story.append(Paragraph("E[θ_k] = α_k / (α_k + β_k),    Var(θ_k) = (α_k · β_k) / [ (α_k + β_k)² (α_k + β_k + 1) ]", formula_style))
    story.append(Paragraph("Thompson Sampling Decision Rule:", h2_style))
    story.append(Paragraph("At each notification trigger, sample θ̃_k ~ Beta(α_k, β_k) for all candidate arms k, and select:", body_style))
    story.append(Paragraph("k* = argmax_k θ̃_k", formula_style))
    story.append(Paragraph(
        "<b>Asymptotic Regret Bounds:</b> Thompson Sampling matches the theoretical Lai-Robbins lower bound for cumulative regret: "
        "E[R(T)] = ∑ (Δ_k / KL(θ_k, θ*)) ln T + o(ln T). Sub-optimal arms are exponentially damped while top-performing arms receive traffic.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Production Alternatives:</b> Contextual Bandits (<b>LinUCB</b> or <b>Disjoint Linear Models</b>) where expected reward is a function of "
        "user feature vector: E[r | x] = xᵀ θ. In non-stationary environments, exponential discount decay (γ = 0.95) prevents posterior freeze.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # SECTION 8: Statistical Testing
    story.append(Paragraph("8. Statistical Testing & Significance Verification", h1_style))
    story.append(Paragraph(
        "<b>Implementation (<code>src/learning/statistical_testing.py</code>):</b> Rigorous verification layer ensuring templates are only declared "
        "winners when supported by both frequentist and Bayesian criteria. Controls Family-Wise Error Rate (FWER) under multiple testing.",
        body_style
    ))
    story.append(Paragraph("Testing Protocols & Multiple Hypothesis Control:", h2_style))
    story.append(Paragraph("Bonferroni Correction over m candidate template hypothesis tests:", body_style))
    story.append(Paragraph("α_adjusted = α / m = 0.05 / m", formula_style))
    story.append(Paragraph("Cohen's h Effect Size for Proportions:", h2_style))
    story.append(Paragraph("h = 2 arcsin( √p₁ ) - 2 arcsin( √p₂ )   (|h| ≥ 0.2 small, |h| ≥ 0.5 medium)", formula_style))
    story.append(Paragraph("Bayesian Probability of Superiority:", h2_style))
    story.append(Paragraph("P(θ_A > θ_B | data) = ∫ f(θ_B | α_B, β_B) [ ∫_{θ_B}¹ f(θ_A | α_A, β_A) dθ_A ] dθ_B", formula_style))
    story.append(Paragraph(
        "<b>Strong Winner Classification:</b> A template is labeled <code>STRONG_WINNER</code> if and only if: "
        "p_frequentist &lt; 0.05 (Bonferroni adjusted) AND P_Bayesian(θ_A &gt; θ_control) &gt; 0.95.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # SECTION 9: NLP Optimization
    story.append(Paragraph("9. NLP Template Optimization & Token Attribution", h1_style))
    story.append(Paragraph(
        "<b>Implementation (<code>src/communication/nlp_template_optimizer.py</code>):</b> Performs token-level and n-gram TF-IDF vectorization "
        "across winning vs. suppressed notifications. Measures statistical correlation between specific vocabulary (e.g., 'hurry', 'miss out', 'streak') "
        "and negative outcomes (uninstalls, opt-outs) to enforce safety guardrails and feed copy generation in Iteration 1.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # SECTION 10: Interview Defense Matrix
    story.append(PageBreak())
    story.append(Paragraph("10. Master Interview Defense & Scenario Matrix", h1_style))
    story.append(Paragraph(
        "This reference matrix prepares you for the toughest questions an ML / System Design interviewer will ask regarding Project Aurora.",
        body_style
    ))
    story.append(Spacer(1, 6))

    qa_matrix = [
        [Paragraph("Interview Question", table_cell_header), Paragraph("Underlying Trap / Concept", table_cell_header), Paragraph("Winning Architectural Answer", table_cell_header)],
        [
            Paragraph("<b>Q1: Why did your churn model achieve AUC 0.44? Isn't that worse than random?</b>", table_cell),
            Paragraph("Testing if you understand AUC and whether you falsify or spin bad results.", table_cell),
            Paragraph("<b>Direct & honest:</b> Yes, AUC 0.44 is worse than a coin flip on this small synthetic dataset. In earlier versions, target leakage artificially yielded AUC 1.0. We removed the circular features to maintain integrity. On this synthetic set, feature-to-churn signal was noise, so our evaluation focused on the engagement regressor and bandit loops. In production, we would enforce an observation window (e.g. 30 days) and separate prediction window (14 days) to build clean labels.", table_cell)
        ],
        [
            Paragraph("<b>Q2: Why use Thompson Sampling instead of traditional A/B testing?</b>", table_cell),
            Paragraph("Understanding the exploration vs. exploitation tradeoff and regret minimization.", table_cell),
            Paragraph("Fixed-horizon A/B testing splits traffic 50/50 until p &lt; 0.05, incurring heavy opportunity cost by exposing users to inferior templates. Thompson Sampling is asymptotically optimal for cumulative regret (Lai-Robbins bound). It shifts traffic dynamically to winning templates as Bayesian evidence accumulates, maximizing user engagement while exploring.", table_cell)
        ],
        [
            Paragraph("<b>Q3: How would you scale the segmentation engine to 50 million users?</b>", table_cell),
            Paragraph("Recognizing algorithmic complexity: Agglomerative clustering is O(N²) memory.", table_cell),
            Paragraph("Agglomerative clustering distance matrices require O(N²) RAM and cannot run on 50M rows in memory. In production, I would use a two-tier approach: train a vector quantizer / BIRCH or K-Means on a representative distributed sample using Apache Spark MLlib, or project features via UMAP and assign users via approximate nearest neighbor (ANN) trees like ScaNN or FAISS in O(log N) time.", table_cell)
        ],
        [
            Paragraph("<b>Q4: How do you prevent notifications from annoying users into uninstalling?</b>", table_cell),
            Paragraph("System safety guardrails and multi-objective optimization.", table_cell),
            Paragraph("We incorporate hard frequency caps (max 1-2 nudges/day) combined with survival hazard triggers. Crucially, we enforce an <b>opt-out circuit breaker</b>: if a segment's uninstall rate exceeds 2%, frequency is automatically throttled by 2/day and high-urgency copy is suppressed. In production, we optimize a composite utility: U = E[CTR] - w · E[Uninstall].", table_cell)
        ],
        [
            Paragraph("<b>Q5: Why did you use Survival Analysis instead of linear regression for timing?</b>", table_cell),
            Paragraph("Statistical rigor regarding censored observational data.", table_cell),
            Paragraph("Event observation in notifications is heavily <b>right-censored</b>: if a user hasn't returned after 4 days, their time-to-return is ≥4 days, not 4 days. Standard regression either deletes censored users (selection bias) or imputes arbitrary cutoffs. Kaplan-Meier and hazard modeling properly account for censored lifespans without parameter distortion.", table_cell)
        ],
        [
            Paragraph("<b>Q6: What if user preferences change over time (non-stationary environment)?</b>", table_cell),
            Paragraph("Bandit lock-in problem due to posterior variance shrinking.", table_cell),
            Paragraph("Standard Thompson Sampling can over-exploit early winners because α + β grows infinitely, reducing variance to zero. To handle non-stationarity, we apply an exponential decay factor γ ∈ (0.9, 0.98) to the Beta parameters: α_(t+1) = γ·α_t + s_t. This bounds total pseudo-counts and preserves exploration flexibility.", table_cell)
        ]
    ]

    t_qa = Table(qa_matrix, colWidths=[120, 114, 270])
    t_qa.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_qa)
    story.append(Spacer(1, 14))

    # SECTION 11: System Optimization Summary
    story.append(Paragraph("11. End-to-End System Optimization Summary", h1_style))
    story.append(Paragraph(
        "To summarize for an architectural interviewer, Project Aurora is not a collection of isolated scripts; "
        "it is a mathematically unified pipeline where each stage feeds an explicit objective function:",
        body_style
    ))
    
    opt_summary_data = [
        [Paragraph("System Block", table_cell_header), Paragraph("Mathematical Objective", table_cell_header), Paragraph("Algorithmic Technique", table_cell_header)],
        [Paragraph("User Clustering", table_cell_bold), Paragraph("min ∑ ||xᵢ - μ_k||² (Minimize intra-cluster variance)", table_cell), Paragraph("Ward's Linkage + Silhouette Sweep", table_cell)],
        [Paragraph("Propensity Modeling", table_cell_bold), Paragraph("min ∑ l(yᵢ, ŷᵢ) + γT + 0.5λ∑wⱼ² (Regularized Risk)", table_cell), Paragraph("XGBoost + LightGBM 5-Fold CV", table_cell)],
        [Paragraph("Send-Time Optimizer", table_cell_bold), Paragraph("max λ(t) (Trigger at maximum hazard of drop-off)", table_cell), Paragraph("Kaplan-Meier Product-Limit Estimator", table_cell)],
        [Paragraph("Message Bandits", table_cell_bold), Paragraph("min E[R(T)] (Minimize cumulative opportunity regret)", table_cell), Paragraph("Thompson Sampling (Beta-Bernoulli)", table_cell)],
        [Paragraph("Statistical Verifier", table_cell_bold), Paragraph("Control FWER ≤ α under m comparisons", table_cell), Paragraph("Bonferroni + Bayesian Credible Posterior", table_cell)],
    ]
    t_opt = Table(opt_summary_data, colWidths=[120, 204, 180])
    t_opt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_opt)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")

if __name__ == "__main__":
    output_filename = "Project_Aurora_Technical_Architecture_and_Interview_Guide.pdf"
    build_pdf(output_filename)
