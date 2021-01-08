import numpy as np
from wordDimensionProject.w2v_utils import *


# words : 사전의 단어들
# word_to_vec_map : 사전 : 각 어휘별로 차원벡터를 담고 있다
words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    distance = 0.0

    # Compute the dot product between u and v (≈1 line)
    dot = np.matmul(u.T, v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(u * u, axis=0))

    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(v * v, axis=0))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity


father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ", cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ", cosine_similarity(france - paris, rome - italy))


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """

    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()


    # Get the word embeddings v_a, v_b and v_c (≈1-3 lines)
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    ### END CODE HERE ###

    words = word_to_vec_map.keys()
    max_cosine_sim = -100  # Initialize max_cosine_sim to a large negative number
    best_word = None  # Initialize best_word with None, it will help keep track of the word to output

    # loop over the whole word vector set
    for w in words:
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_b, word_c]:
            continue

        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  (≈1 line)
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        # If the cosine_sim is more than the max_cosine_sim seen so far,
        # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word


triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'),
                 ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print('{} -> {} :: {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_map)))


# g벡터, gender축을 구할 때, 아래와 같이 하나의 쌍만으로 차원축을 구하지 않고
# 여러 단어의 벡터차를 구해서, 평균을 내는 방식으로 구하면 좀 더 정확한 차원축을 얻을 수 있다
g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g)
print('List of names and their similarities with constructed vector:')

# girls and boys name
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
for w in name_list:
    print(w, cosine_similarity(word_to_vec_map[w], g))

print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior', 'doctor', 'tree', 'receptionist',
             'technology', 'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print(w, cosine_similarity(word_to_vec_map[w], g))


# '엔지니어' 와 같은 특정 단어가 gender 차원축에 수직이 되도록 벡터공간 재정렬
# '언제니어'라는 어휘가 gender라는 차원속성에서 중성이 되도록 재정렬한다
def neutralize(word, g, word_to_vec_map):
    e = word_to_vec_map[word]

    # 정사영 벡터 계산
    e_biascomponent = (np.matmul(e.T, g) / (np.linalg.norm(g, axis=0, keepdims=True) ** 2)) * g

    # 벡터 뉴트럴라이즈
    e_debiased = e - e_biascomponent

    return e_debiased


e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ",
      cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))


# 위의 작업으로, 과학자라는 단어가 gender에 대해 수직한 벡터가 되었다고 하자(중성화 완료)
# 이제 girl, boy 두 단어가 존재한다고 하자
# 이 두 단어는 자체적으로 성별을 내포한 단어이기 때문에, 중성화시켜서는 안된다
# 이제 boy, girl 두 단어가 과학자라는 단어와 같은 거리를 유지할 수 있다면, 과학자라는 단어가 성별에 중성화될 수 있다
# 그러나, boy, girl 두 단어가 과학자라는 단어와 이루는 거리가 서로 다를 수 있다
# 따라서, boy, girl이라는 두 단어가 gender에 수직인 축들과 이루는 거리가 모두 동일하도록 조절하자
# 젠더축에서 거리가 동일한 두 베거를 ㅁ
def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
    mu = (e_w1 + e_w2) / 2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
    mu_B = (np.matmul(mu.T, bias_axis) / (np.linalg.norm(bias_axis, axis=0, keepdims=True) ** 2)) * bias_axis
    mu_orth = mu - mu_B

    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
    e_w1B = (np.matmul(e_w1.T, bias_axis) / (np.linalg.norm(bias_axis, axis=0, keepdims=True) ** 2)) * bias_axis
    e_w2B = (np.matmul(e_w2.T, bias_axis) / (np.linalg.norm(bias_axis, axis=0, keepdims=True) ** 2)) * bias_axis

    # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
    corrected_e_w1B = np.sqrt(np.abs(1 - np.linalg.norm(mu_orth) ** 2)) * ((e_w1B - mu_B) / np.linalg.norm((e_w1 - mu_orth) - mu_B, axis=0, keepdims=True))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.linalg.norm(mu_orth) ** 2)) * ((e_w2B - mu_B) / np.linalg.norm((e_w2 - mu_orth) - mu_B, axis=0, keepdims=True))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2


print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
