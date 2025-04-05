# import spacy
# from collections import Counter
# import random

# class ObjectiveTest:

#     def __init__(self, data, noOfQues):
#         self.nlp = spacy.load("en_core_web_sm")
#         self.text = data
#         self.noOfQues = int(noOfQues)

#     def generate_mcqs(self):
#         # Process the text with spaCy
#         doc = self.nlp(self.text)

#         # Extract sentences from the text
#         sentences = [sent.text for sent in doc.sents]

#         # Randomly select sentences to form questions
#         selected_sentences = random.sample(sentences, min(self.noOfQues, len(sentences)))

#         # Initialize list to store generated MCQs
#         mcqs = []

#         # Generate MCQs for each selected sentence
#         for sentence in selected_sentences:
#             # Process the sentence with spaCy
#             sent_doc = self.nlp(sentence)

#             # Extract entities (nouns) from the sentence
#             nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]

#             # Ensure there are enough nouns to generate MCQs
#             if len(nouns) < 2:
#                 continue

#             # Count the occurrence of each noun
#             noun_counts = Counter(nouns)

#             # Select the most common noun as the subject of the question
#             if noun_counts:
#                 subject = noun_counts.most_common(1)[0][0]

#                 # Generate the question stem
#                 question_stem = sentence.replace(subject, "_______")

#                 # Generate answer choices
#                 answer_choices = [subject]

#                 # Add some random words from the text as distractors
#                 for _ in range(3):
#                     distractor = random.choice(list(set(nouns) - set([subject])))
#                     answer_choices.append(distractor)

#                 # Shuffle the answer choices
#                 random.shuffle(answer_choices)

#                 # Enumerate the choices
#                 enumerated_choices = [(chr(65 + i), choice) for i, choice in enumerate(answer_choices)]

#                 # Append the generated MCQ to the list
#                 correct_answer = chr(65 + answer_choices.index(subject))  # Convert index to letter (A, B, C, D)
#                 mcqs.append((question_stem, enumerated_choices, correct_answer))

#         return mcqs

import random
from collections import Counter

class ObjectiveTest:

    def __init__(self, data, noOfQues):
        self.text = data
        self.noOfQues = int(noOfQues)

    def generate_mcqs(self):
        # Simple sentence splitting by common punctuation
        sentences = [s.strip() for s in self.text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        # Filter out sentences that are too short
        sentences = [s for s in sentences if len(s.split()) > 10]
        
        # Randomly select sentences to form questions
        if len(sentences) < self.noOfQues:
            self.noOfQues = len(sentences)
            
        selected_sentences = random.sample(sentences, self.noOfQues) if sentences else []
        
        # Initialize list to store generated MCQs
        mcqs = []

        # Generate MCQs for each selected sentence
        for sentence in selected_sentences:
            # Simple word extraction - not as accurate as NLP, but works as placeholder
            words = sentence.split()
            
            # Try to identify important words (longer words might be more important)
            potential_keywords = [word for word in words if len(word) > 5 and word.isalpha()]
            
            # If we don't have enough potential keywords, use all words
            if len(potential_keywords) < 4:
                potential_keywords = [word for word in words if word.isalpha()]
                
            # Skip if still not enough words
            if len(potential_keywords) < 4:
                continue
                
            # Get word frequency
            word_counts = Counter(potential_keywords)
            
            # If we have any words to work with
            if word_counts:
                # Try to select a reasonably common word
                subject = word_counts.most_common()[0][0]
                
                # Generate the question stem by replacing the subject
                question_stem = sentence.replace(subject, "_______", 1)
                
                # Generate unique answer choices
                answer_choices = [subject]  # Start with the correct answer
                
                # Add distractors from other words in the text
                available_distractors = [word for word in potential_keywords if word != subject]
                
                # Add distractors if there are enough available
                if len(available_distractors) >= 3:
                    for _ in range(3):
                        if available_distractors:
                            distractor = random.choice(available_distractors)
                            answer_choices.append(distractor)
                            available_distractors.remove(distractor)
                
                # Ensure we have exactly 4 choices
                while len(answer_choices) < 4:
                    answer_choices.append(f"Option {len(answer_choices) + 1}")
                
                # Shuffle the choices
                random.shuffle(answer_choices)
                
                # Enumerate the choices
                enumerated_choices = [(chr(65 + i), choice) for i, choice in enumerate(answer_choices)]
                
                # Append the generated MCQ to the list
                correct_answer = chr(65 + answer_choices.index(subject))  # Convert index to letter (A, B, C, D)
                mcqs.append((question_stem, enumerated_choices, correct_answer))
                
        return mcqs
