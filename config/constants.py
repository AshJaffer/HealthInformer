"""PubMed search queries and topic categories that define the corpus scope.

Each query maps to a human-readable category used for metadata tagging.
These queries are intentionally broad to capture a wide range of health
literacy topics relevant to the general public.
"""

# Maps a PubMed search query → a short topic category label.
# The label travels with every chunk for filtering and citation display.
PUBMED_QUERIES: dict[str, str] = {
    # Chronic diseases
    "type 2 diabetes management": "Diabetes",
    "hypertension treatment guidelines": "Hypertension",
    "asthma management adults": "Asthma",
    "heart failure patient education": "Heart Failure",
    "chronic kidney disease prevention": "Kidney Disease",

    # Cancer screening & prevention
    "breast cancer screening recommendations": "Cancer Screening",
    "colorectal cancer prevention": "Cancer Prevention",
    "lung cancer risk factors smoking": "Cancer Risk",

    # Mental health
    "depression treatment options": "Mental Health",
    "anxiety disorders cognitive behavioral therapy": "Mental Health",
    "sleep disorders insomnia management": "Sleep Health",

    # Nutrition & lifestyle
    "healthy diet cardiovascular health": "Nutrition",
    "physical activity guidelines adults": "Exercise",
    "obesity weight management interventions": "Weight Management",

    # Infectious disease & vaccines
    "influenza vaccination effectiveness": "Vaccines",
    "COVID-19 long term effects": "COVID-19",
    "antibiotic resistance patient education": "Antibiotics",

    # Women's & reproductive health
    "prenatal care guidelines": "Prenatal Health",
    "menopause symptom management": "Women's Health",

    # Aging & musculoskeletal
    "osteoporosis prevention treatment": "Bone Health",
    "fall prevention elderly": "Geriatrics",
}
