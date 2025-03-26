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

import spacy
from collections import Counter
import random

class ObjectiveTest:

    def __init__(self, data, noOfQues):
        self.nlp = spacy.load("en_core_web_sm")
        self.text = data
        self.noOfQues = int(noOfQues)

    def generate_mcqs(self):
        # Process the text with spaCy
        doc = self.nlp(self.text)

        # Extract sentences from the text
        sentences = [sent.text for sent in doc.sents]

        # Randomly select sentences to form questions
        selected_sentences = random.sample(sentences, min(self.noOfQues, len(sentences)))

        # Initialize list to store generated MCQs
        mcqs = []

        # Generate MCQs for each selected sentence
        for sentence in selected_sentences:
            # Process the sentence with spaCy
            sent_doc = self.nlp(sentence)

            # Extract entities (nouns) from the sentence
            nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]

            # Ensure there are enough nouns to generate MCQs
            if len(nouns) < 2:
                continue

            # Count the occurrence of each noun
            noun_counts = Counter(nouns)

            # Select the most common noun as the subject of the question
            if noun_counts:
                subject = noun_counts.most_common(1)[0][0]

                # Generate the question stem by replacing the first occurrence of the subject with "_______"
                question_stem = sentence.replace(subject, "_______", 1)

                # Generate unique answer choices
                answer_choices = set([subject])  # Start with the correct answer
                available_distractors = list(set(nouns) - set(answer_choices))
                
                # Add distractors if there are enough available
                if len(available_distractors) >= 3:
                    while len(answer_choices) < 4:
                        distractor = random.choice(available_distractors)
                        answer_choices.add(distractor)
                        available_distractors.remove(distractor)  # Ensure no duplicate distractors
                else:
                    # If there aren't enough unique distractors, use whatever is available
                    answer_choices.update(available_distractors)

                # Convert set to list and shuffle the choices
                answer_choices = list(answer_choices)
                random.shuffle(answer_choices)

                # Ensure we have exactly 4 choices
                if len(answer_choices) < 4:
                    # If we don't have enough choices, fill the rest with placeholders
                    while len(answer_choices) < 4:
                        answer_choices.append("Placeholder")
                
                # Enumerate the choices
                enumerated_choices = [(chr(65 + i), choice) for i, choice in enumerate(answer_choices)]

                # Append the generated MCQ to the list
                correct_answer = chr(65 + answer_choices.index(subject))  # Convert index to letter (A, B, C, D)
                mcqs.append((question_stem, enumerated_choices, correct_answer))

        return mcqs
