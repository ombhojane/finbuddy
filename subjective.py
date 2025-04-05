# import numpy as np
# import nltk as nlp

# class SubjectiveTest:

#     def __init__(self, data, noOfQues):
#         self.question_pattern = [
#             "Explain in detail ",
#             "Define ",
#             "Write a short note on ",
#             "What do you mean by "
#         ]
#         self.grammar = r"""
#             CHUNK: {<NN>+<IN|DT>*<NN>+}
#             {<NN>+<IN|DT>*<NNP>+}
#             {<NNP>+<NNS>*}
#         """
#         self.summary = data
#         self.noOfQues = noOfQues
    
#     @staticmethod
#     def word_tokenizer(sequence):
#         word_tokens = []
#         for sent in nlp.sent_tokenize(sequence):
#             word_tokens.extend(nlp.word_tokenize(sent))
#         return word_tokens
    
#     @staticmethod
#     def create_vector(answer_tokens, tokens):
#         return np.array([1 if tok in answer_tokens else 0 for tok in tokens])
    
#     @staticmethod
#     def cosine_similarity_score(vector1, vector2):
#         def vector_value(vector):
#             return np.sqrt(np.sum(np.square(vector)))
#         v1 = vector_value(vector1)
#         v2 = vector_value(vector2)
#         v1_v2 = np.dot(vector1, vector2)
#         return (v1_v2 / (v1 * v2)) * 100
    
#     def generate_test(self):
#         sentences = nlp.sent_tokenize(self.summary)
#         cp = nlp.RegexpParser(self.grammar)
#         question_answer_dict = {}
        
#         for sentence in sentences:
#             tagged_words = nlp.pos_tag(nlp.word_tokenize(sentence))
#             tree = cp.parse(tagged_words)
#             for subtree in tree.subtrees():
#                 if subtree.label() == "CHUNK":
#                     temp = " ".join([sub[0] for sub in subtree]).strip().upper()
#                     if temp not in question_answer_dict and len(nlp.word_tokenize(sentence)) > 20:
#                         question_answer_dict[temp] = sentence
        
#         keyword_list = list(question_answer_dict.keys())
#         question_answer = []
#         for _ in range(int(self.noOfQues)):
#             if keyword_list:
#                 rand_num = np.random.randint(0, len(keyword_list))
#                 selected_key = keyword_list[rand_num]
#                 answer = question_answer_dict[selected_key]
#                 rand_num %= 4
#                 question = self.question_pattern[rand_num] + selected_key + "."
#                 question_answer.append({"Question": question, "Answer": answer})
        
#         que, ans = [], []
#         while len(que) < int(self.noOfQues) and question_answer:
#             rand_num = np.random.randint(0, len(question_answer))
#             if question_answer[rand_num]["Question"] not in que:
#                 que.append(question_answer[rand_num]["Question"])
#                 ans.append(question_answer[rand_num]["Answer"])
        
#         return que, ans

import random

class SubjectiveTest:

    def __init__(self, data, noOfQues):
        self.question_pattern = [
            "Explain in detail ",
            "Define ",
            "Write a short note on ",
            "What do you mean by "
        ]
        self.summary = data
        self.noOfQues = noOfQues

    def generate_test(self):
        # Simple sentence splitting by common punctuation
        sentences = [s.strip() for s in self.summary.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        # Filter out sentences that are too short
        sentences = [s for s in sentences if len(s.split()) > 10]
        
        # Generate questions and answers
        que, ans = [], []
        
        if len(sentences) < self.noOfQues:
            self.noOfQues = len(sentences)
        
        # Just use sentence topics (first few words) as question topics
        selected_sentences = random.sample(sentences, self.noOfQues) if sentences else []
        
        for sentence in selected_sentences:
            words = sentence.split()
            # Use first 3-5 words as the topic
            topic_length = min(len(words), random.randint(3, 5))
            topic = " ".join(words[:topic_length]).upper()
            
            # Select a random question pattern
            pattern_idx = random.randint(0, len(self.question_pattern)-1)
            question = self.question_pattern[pattern_idx] + topic + "."
            
            que.append(question)
            ans.append(sentence)
            
        return que, ans


