"""Curated test questions spanning all topic categories.

These are our "gold" test set for RAGAS evaluation. Each question includes
an optional ground_truth answer sketch (used by context_recall) and the
expected topic category for analysis breakdowns.
"""

# Each entry: (question, category, ground_truth_sketch)
# ground_truth_sketch is a brief expected answer — not exhaustive, just
# enough for RAGAS context_recall to check if retrieval covered the key facts.
TEST_QUESTIONS: list[tuple[str, str, str]] = [
    # ── Diabetes ────────────────────────────────────────────────────────
    (
        "What are the warning signs of type 2 diabetes?",
        "Diabetes",
        "Common warning signs include increased thirst, frequent urination, "
        "unexplained weight loss, fatigue, blurred vision, and slow-healing wounds.",
    ),
    (
        "How is type 2 diabetes typically managed?",
        "Diabetes",
        "Management includes lifestyle changes (diet, exercise), blood sugar monitoring, "
        "oral medications like metformin, and sometimes insulin therapy.",
    ),

    # ── Hypertension ────────────────────────────────────────────────────
    (
        "What lifestyle changes help lower blood pressure?",
        "Hypertension",
        "Reducing sodium intake, regular exercise, maintaining healthy weight, "
        "limiting alcohol, managing stress, and eating a DASH-style diet.",
    ),
    (
        "What are the risks of untreated high blood pressure?",
        "Hypertension",
        "Untreated hypertension increases risk of heart attack, stroke, kidney damage, "
        "vision loss, and heart failure.",
    ),

    # ── Asthma ──────────────────────────────────────────────────────────
    (
        "What triggers can worsen asthma symptoms in adults?",
        "Asthma",
        "Common triggers include allergens, air pollution, respiratory infections, "
        "exercise, cold air, smoke, and stress.",
    ),

    # ── Heart Failure ───────────────────────────────────────────────────
    (
        "What are the early symptoms of heart failure?",
        "Heart Failure",
        "Early symptoms include shortness of breath, fatigue, swelling in legs and ankles, "
        "rapid or irregular heartbeat, and reduced exercise tolerance.",
    ),

    # ── Kidney Disease ──────────────────────────────────────────────────
    (
        "How can chronic kidney disease be prevented?",
        "Kidney Disease",
        "Prevention includes controlling blood pressure and blood sugar, maintaining healthy weight, "
        "avoiding excessive NSAID use, staying hydrated, and not smoking.",
    ),

    # ── Cancer Screening ────────────────────────────────────────────────
    (
        "At what age should breast cancer screening begin?",
        "Cancer Screening",
        "Guidelines generally recommend mammography screening starting at age 40-50, "
        "depending on risk factors and the guideline organization.",
    ),
    (
        "What are the recommended screenings for colorectal cancer?",
        "Cancer Prevention",
        "Recommended screenings include colonoscopy, stool-based tests (FIT, gFOBT), "
        "and CT colonography, typically starting at age 45.",
    ),

    # ── Cancer Risk ─────────────────────────────────────────────────────
    (
        "How does smoking increase the risk of lung cancer?",
        "Cancer Risk",
        "Smoking damages lung cells through carcinogens in tobacco smoke, causing DNA mutations "
        "that accumulate over time and lead to uncontrolled cell growth.",
    ),

    # ── Mental Health ───────────────────────────────────────────────────
    (
        "What are evidence-based treatments for depression?",
        "Mental Health",
        "Evidence-based treatments include cognitive behavioral therapy (CBT), "
        "antidepressant medications (SSRIs, SNRIs), exercise, and combination therapy.",
    ),
    (
        "How effective is cognitive behavioral therapy for anxiety?",
        "Mental Health",
        "CBT is considered a first-line treatment for anxiety disorders with strong evidence "
        "for its effectiveness, often comparable to or better than medication alone.",
    ),

    # ── Sleep Health ────────────────────────bedeutung────────────────────────
    (
        "What are recommended treatments for chronic insomnia?",
        "Sleep Health",
        "Recommended treatments include cognitive behavioral therapy for insomnia (CBT-I), "
        "sleep hygiene practices, and in some cases short-term medication.",
    ),

    # ── Nutrition ───────────────────────────────────────────────────────
    (
        "What dietary patterns are best for heart health?",
        "Nutrition",
        "Heart-healthy diets include the Mediterranean diet, DASH diet, and plant-based diets "
        "rich in fruits, vegetables, whole grains, lean protein, and healthy fats.",
    ),
    (
        "How does diet affect cardiovascular disease risk?",
        "Nutrition",
        "Diet affects cardiovascular risk through blood pressure, cholesterol levels, "
        "inflammation, blood sugar control, and body weight.",
    ),

    # ── Exercise ────────────────────────────────────────────────────────
    (
        "How much physical activity do adults need per week?",
        "Exercise",
        "Adults should get at least 150 minutes of moderate-intensity or 75 minutes of "
        "vigorous-intensity aerobic activity per week, plus muscle-strengthening activities.",
    ),

    # ── Weight Management ───────────────────────────────────────────────
    (
        "What interventions are effective for obesity management?",
        "Weight Management",
        "Effective interventions include caloric restriction, increased physical activity, "
        "behavioral counseling, and in some cases medication or bariatric surgery.",
    ),

    # ── Vaccines ────────────────────────────────────────────────────────
    (
        "How effective is the annual influenza vaccine?",
        "Vaccines",
        "Influenza vaccine effectiveness varies by season (typically 40-60%), "
        "but vaccination reduces severity of illness and risk of hospitalization.",
    ),
    (
        "Who should get the flu vaccine each year?",
        "Vaccines",
        "Everyone 6 months and older should receive annual flu vaccination, "
        "especially high-risk groups like elderly, pregnant women, and immunocompromised individuals.",
    ),

    # ── COVID-19 ────────────────────────────────────────────────────────
    (
        "What are the long-term effects of COVID-19?",
        "COVID-19",
        "Long COVID symptoms include fatigue, brain fog, shortness of breath, "
        "chest pain, joint pain, and cardiovascular complications.",
    ),

    # ── Antibiotics ─────────────────────────────────────────────────────
    (
        "Why is antibiotic resistance a growing concern?",
        "Antibiotics",
        "Overuse and misuse of antibiotics drive resistance, making infections harder to treat, "
        "increasing healthcare costs, and raising mortality risk.",
    ),

    # ── Prenatal Health ─────────────────────────────────────────────────
    (
        "What are the key components of prenatal care?",
        "Prenatal Health",
        "Key components include regular check-ups, nutritional guidance, folic acid supplementation, "
        "screening tests, monitoring fetal development, and managing risk factors.",
    ),

    # ── Women's Health ──────────────────────────────────────────────────
    (
        "What are common symptoms of menopause?",
        "Women's Health",
        "Common symptoms include hot flashes, night sweats, mood changes, sleep disturbances, "
        "vaginal dryness, and changes in bone density.",
    ),
    (
        "What treatment options exist for menopause symptoms?",
        "Women's Health",
        "Options include hormone replacement therapy (HRT), non-hormonal medications, "
        "lifestyle modifications, and cognitive behavioral therapy.",
    ),

    # ── Bone Health ─────────────────────────────────────────────────────
    (
        "How can osteoporosis be prevented?",
        "Bone Health",
        "Prevention includes adequate calcium and vitamin D intake, weight-bearing exercise, "
        "avoiding smoking and excessive alcohol, and bone density screening.",
    ),

    # ── Geriatrics ──────────────────────────────────────────────────────
    (
        "What strategies help prevent falls in elderly adults?",
        "Geriatrics",
        "Strategies include exercise programs for balance and strength, home safety modifications, "
        "medication review, vision checks, and assistive devices.",
    ),

    # ── Cross-cutting questions ─────────────────────────────────────────
    (
        "What is the relationship between obesity and type 2 diabetes?",
        "Diabetes",
        "Obesity is a major risk factor for type 2 diabetes due to insulin resistance. "
        "Excess body fat, especially visceral fat, impairs insulin signaling.",
    ),
    (
        "How does exercise benefit mental health?",
        "Mental Health",
        "Exercise reduces symptoms of depression and anxiety through endorphin release, "
        "improved sleep, stress reduction, and increased self-efficacy.",
    ),
    (
        "What vaccines are recommended for adults over 65?",
        "Vaccines",
        "Recommended vaccines include influenza (high-dose), pneumococcal, shingles (Shingrix), "
        "Tdap booster, and COVID-19 boosters.",
    ),
    (
        "How does chronic stress affect physical health?",
        "Mental Health",
        "Chronic stress increases risk of cardiovascular disease, weakens immune function, "
        "disrupts sleep, raises blood pressure, and can worsen chronic conditions.",
    ),
]
