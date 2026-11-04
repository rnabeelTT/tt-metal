# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Diverse prompts for routing capture (static expert plan) and held-out evaluation. TRAIN builds the plan; HELDOUT is
never used for planning."""

TRAIN = [
    # multilingual
    "用中文解释一下什么是区块链，以及它为什么被认为是安全的。",
    "Escribe un correo formal en español solicitando una reunión con el director de recursos humanos.",
    "Erkläre auf Deutsch, wie eine Wärmepumpe funktioniert und warum sie effizient ist.",
    "Rédige en français un court poème sur l'automne à Paris.",
    "日本語で、東京で一日観光するためのおすすめプランを教えてください。",
    "اشرح باللغة العربية فوائد القراءة اليومية للأطفال.",
    "हिंदी में समझाइए कि सौर ऊर्जा कैसे काम करती है।",
    "Объясни по-русски, почему небо голубое.",
    "Spiega in italiano la differenza tra il caffè espresso e il caffè americano.",
    "Explique em português como funciona o sistema imunológico humano.",
    "한국어로 김치를 만드는 기본적인 방법을 설명해 주세요.",
    "Translate into German and Japanese: 'The meeting has been moved to Thursday afternoon because of the holiday.'",
    # code
    "Write a Go HTTP server with a /health endpoint and graceful shutdown on SIGTERM.",
    "Implement quicksort in C++ with a median-of-three pivot and explain its complexity.",
    "Write a TypeScript React hook that debounces a value, with usage example.",
    "Write a Java class for a bank account with deposit, withdraw, and thread safety.",
    "Explain this Python error and fix it: 'RecursionError: maximum recursion depth exceeded' in a fibonacci function.",
    "Write a SQL query that finds the top 3 customers by revenue per region using window functions.",
    "Write a Dockerfile for a Python FastAPI app with a multi-stage build and a non-root user.",
    "Write a Haskell function to compute the n-th prime and explain laziness in it.",
    "Write a CUDA kernel for vector addition and the host code to launch it.",
    "Convert this regex to plain English and give test cases: ^(?=.*[A-Z])(?=.*\\d)[A-Za-z\\d@$!%*?&]{8,}$",
    # math / reasoning
    "Prove that the square root of 2 is irrational.",
    "A bag has 5 red and 7 blue balls. Two are drawn without replacement. What is the probability both are blue? Show work.",
    "Solve the system: 3x + 2y = 16, 5x - y = 18. Explain each step.",
    "Explain Bayes' theorem with a medical test example and concrete numbers.",
    "If a clock shows 3:15, what is the angle between the hour and minute hands? Explain.",
    "Compute the derivative of x^3 * ln(x) and explain the product rule.",
    # knowledge / writing / chat
    "Summarize the plot of Romeo and Juliet in five bullet points.",
    "What are the pros and cons of nuclear energy? Give a balanced answer.",
    "Write a cover letter for a junior data analyst position at a retail company.",
    "Give me tips to improve my sleep schedule as a night-shift nurse.",
    "Explain the difference between a virus and a bacterium to a 10-year-old.",
    "Write a haiku sequence about a city in the rain.",
    "Describe the main causes of World War I in about 200 words.",
    "How do I negotiate a higher salary? Give a short script.",
    "Plan a 3-day budget trip to Lisbon with daily itineraries.",
    "Explain how the stock market works to a complete beginner.",
    "Write a product description for noise-cancelling headphones aimed at commuters.",
    "What is the difference between machine learning and deep learning? Use examples.",
]

HELDOUT = [
    "用中文写一段关于春节传统习俗的介绍。",
    "Écris en français une lettre de motivation pour un poste de développeur web.",
    "Escribe en español una receta de paella valenciana paso a paso.",
    "Write a Rust function that parses a CSV line respecting quoted fields, with tests.",
    "Write a Python script that downloads a web page and counts word frequencies.",
    "Prove that there are infinitely many prime numbers.",
    "Explain how vaccines train the immune system, in simple terms.",
    "Write a short mystery story set on a night train, about 300 words.",
]
