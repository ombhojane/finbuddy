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

import numpy as np
import nltk as nlp
from text_filter import clean_text, preserve_content_length

class SubjectiveTest:

    def __init__(self, data, noOfQues):
        self.question_pattern = [
            "Explain in detail ",
            "Define ",
            "Write a short note on ",
            "What do you mean by "
        ]
        self.grammar = r"""
            CHUNK: {<NN>+<IN|DT>*<NN>+}
            {<NN>+<IN|DT>*<NNP>+}
            {<NNP>+<NNS>*}
        """
        self.summary = data
        self.noOfQues = noOfQues

        # Clean and preserve the content length
        cleaned_text = clean_text(self.summary)
        self.summary = preserve_content_length(self.summary, cleaned_text, retention_factor=0.9)

    @staticmethod
    def word_tokenizer(sequence):
        word_tokens = []
        for sent in nlp.sent_tokenize(sequence):
            word_tokens.extend(nlp.word_tokenize(sent))
        return word_tokens

    @staticmethod
    def create_vector(answer_tokens, tokens):
        return np.array([1 if tok in answer_tokens else 0 for tok in tokens])

    @staticmethod
    def cosine_similarity_score(vector1, vector2):
        def vector_value(vector):
            return np.sqrt(np.sum(np.square(vector)))
        v1 = vector_value(vector1)
        v2 = vector_value(vector2)
        v1_v2 = np.dot(vector1, vector2)
        return (v1_v2 / (v1 * v2)) * 100

    def generate_test(self):
        sentences = nlp.sent_tokenize(self.summary)
        cp = nlp.RegexpParser(self.grammar)
        question_answer_dict = {}

        for sentence in sentences:
            tagged_words = nlp.pos_tag(nlp.word_tokenize(sentence))
            tree = cp.parse(tagged_words)
            for subtree in tree.subtrees():
                if subtree.label() == "CHUNK":
                    chunk_text = " ".join([sub[0] for sub in subtree]).strip().upper()
                    
                    # Ensure chunk text is relevant and not too short
                    if len(chunk_text.split()) > 1 and len(nlp.word_tokenize(sentence)) > 20:
                        question_answer_dict[chunk_text] = sentence
        
        # Generate questions based on the most frequent chunks
        keyword_list = sorted(question_answer_dict.keys(), key=lambda x: len(x.split()), reverse=True)
        question_answer = []
        for _ in range(int(self.noOfQues)):
            if keyword_list:
                rand_num = np.random.randint(0, len(keyword_list))
                selected_key = keyword_list[rand_num]
                answer = question_answer_dict[selected_key]
                rand_num %= len(self.question_pattern)
                question = self.question_pattern[rand_num] + selected_key + "."
                question_answer.append({"Question": question, "Answer": answer})

        # Ensure unique questions and answers
        que, ans = [], []
        unique_questions = set()
        while len(que) < int(self.noOfQues) and question_answer:
            rand_num = np.random.randint(0, len(question_answer))
            question = question_answer[rand_num]["Question"]
            if question not in unique_questions:
                unique_questions.add(question)
                que.append(question)
                ans.append(question_answer[rand_num]["Answer"])

        return que, ans


