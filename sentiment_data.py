"""Общий датасет для sentiment-классификации без дубликатов текстов."""

POSITIVE_TEXTS = [
    "I love this movie, it's fantastic!",
    "Great acting and wonderful plot.",
    "Best film I've seen this year.",
    "Amazing experience, highly recommended.",
    "I really enjoyed this cinema masterpiece.",
    "The cinematography was breathtaking and the story moved me.",
    "A heartwarming tale with brilliant performances from the cast.",
    "This director never disappoints — another stunning achievement.",
    "I laughed and cried; truly an unforgettable movie night.",
    "The soundtrack perfectly complements every emotional scene.",
    "Clever writing keeps you engaged from start to finish.",
    "Every character felt authentic and well developed.",
    "A perfect blend of humor, drama, and suspense.",
    "I would watch this again without hesitation.",
    "The pacing was excellent and the ending was satisfying.",
    "Visually stunning with a compelling narrative arc.",
    "One of the most original films in recent memory.",
    "The lead actor delivered a career-defining performance.",
    "I left the theater feeling inspired and uplifted.",
    "A must-see for anyone who appreciates great storytelling.",
    "The chemistry between the two leads was electric.",
    "Thought-provoking themes handled with sensitivity and depth.",
    "Even the supporting cast shone in their roles.",
    "This movie exceeded all my expectations.",
    "A delightful surprise that I cannot stop recommending.",
    "The dialogue was sharp, witty, and memorable.",
    "Beautifully shot scenes that linger in your mind.",
    "An emotional rollercoaster in the best possible way.",
    "The plot twists were clever without feeling forced.",
    "Simply outstanding — a gem of modern cinema.",
]

NEGATIVE_TEXTS = [
    "Terrible movie, waste of time.",
    "I hated the plot and the acting was bad.",
    "Worst film ever. Don't watch it.",
    "Boring and predictable story.",
    "I didn't like it at all, very disappointing.",
    "The script felt lazy and the pacing dragged endlessly.",
    "Flat performances ruined what could have been a decent idea.",
    "I struggled to stay awake during the second half.",
    "Clichéd characters and zero emotional impact.",
    "The special effects looked cheap and distracting.",
    "A confusing mess with no coherent storyline.",
    "I want those two hours of my life back.",
    "The humor fell flat and the drama felt forced.",
    "Poorly edited scenes made the film hard to follow.",
    "The ending made no sense and left me frustrated.",
    "Overacted and underwritten from beginning to end.",
    "Nothing about this movie felt original or fresh.",
    "The soundtrack was annoying and out of place.",
    "I expected much more given the talented cast.",
    "Shallow characters with no meaningful development.",
    "The plot holes were impossible to ignore.",
    "A tedious experience I would not recommend to anyone.",
    "The dialogue sounded unnatural and cringe-worthy.",
    "Visually dull with uninspired camera work.",
    "The story went nowhere and the tone was inconsistent.",
    "Even the action sequences felt boring and repetitive.",
    "A forgettable film that fails on every level.",
    "The premise was promising but the execution was awful.",
    "I walked out before the credits rolled.",
    "A complete disaster from the opening scene onward.",
]


def get_labeled_data():
    """Возвращает DataFrame-подобный словарь с уникальными текстами и метками."""
    texts = POSITIVE_TEXTS + NEGATIVE_TEXTS
    labels = [1] * len(POSITIVE_TEXTS) + [0] * len(NEGATIVE_TEXTS)
    return {"text": texts, "label": labels}
