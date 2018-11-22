import re
from konlpy.tag import Twitter
import math
import heapq, random
import numpy as np
import pickle

def read_data():
    global data
    data = {}
    path = './data/'
    file_names = ['fam','lov','sch', 'soc', 'topic1_family', 'topic2_school', 'topic3_love', 'topic4_society', 'word_vec']
    file_name_extension = 'pic'
    for file_name in file_names:
        s = path + file_name + '.' + file_name_extension
        with open(s, 'rb') as f:
            data[file_name] = pickle.load(f)
    return data

pos_tagger = Twitter()
def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(str(doc), norm=True, stem=True)]

def word_count(docs):
    topic = [
        set(["엄마/Noun","아빠/Noun","아버지/Noun","어머니/Noun","할머니/Noun","부모님/Noun","동생/Noun","가족/Noun","아들/Noun","집안/Noun","자식/Noun","결혼/Noun","이혼/Noun","사촌/Noun"]),
        set(["선배/Noun","새내기/Noun","후배/Noun","동기/Noun","동아리/Noun","행사/Noun","인사/Noun","술자리/Noun","학생회/Noun","학교/Noun","학년/Noun","입학/Noun","활동/Noun","술/Noun","개강/Noun","밥약/Noun","꼰대/Noun","존댓말/Noun","학번/Noun","학우/Noun","존댓말/Noun","학번/Noun","신입생/Noun"]),
        set(["사랑/Noun","마음/Noun","행복/Noun","감정/Noun","추억/Noun","상처/Noun","이별/Noun","서로/Noun","연애/Noun","벚꽃/Noun","미안/Noun","후회/Noun","마지막/Noun","소중/Noun","미소/Noun","표현/Noun","따뜻/Noun","첫사랑/Noun","웃음/Noun","곰신/Noun","고백/Noun","성격/Noun","사이/Noun","서운/Noun","남자친구/Noun","여자친구/Noun"]),
        set(["사회/Noun","문제/Noun","여성/Noun","이유/Noun","의견/Noun","동성애/Noun","잘못/Noun","종교/Noun","정치/Noun","집단/Noun","혐오/Noun","행위/Noun","차별/Noun","주장/Noun","가치관/Noun","정치/Noun","소수자/Noun","자유/Noun","발언/Noun"])
    ]
    count = [0,0,0,0]    
    for word in docs:
        for i in range(4):
            if word in topic[i]:
                count[i] += 1
                break
    return count ##max값이 2개면;;?

def prepro(s):
    hangul = re.compile('[^ |가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
    result = hangul.sub('', s) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
    return(result)

def dot_product(v1, v2):
    return sum(v1*v2)

def cosine_measure(v1, v2):
    return dot_product(v1, v2) / (math.sqrt(dot_product(v1, v1)) * math.sqrt(dot_product(v2, v2)))

def Djbamboo(sa):
    #75 set {{docs}}
    docs = tokenize(sa)
    
    #77~108 STEP1 : word count & find max position
    max_index = np.argmax(word_count(docs))
    cate = ['fam', 'sch', 'lov', 'soc'][max_index]
    topic = ['topic1_family', 'topic2_school', 'topic3_love', 'topic4_society'][max_index]

    #111 ~ 117 STEP2 : docs to word2vec position
    rex = [s2 for s2 in [prepro(s) for s in docs] if s2!='']

    #119~122
    ind = []
    for r in rex:
        if r in data['word_vec'].keys():
            ind.append(data['word_vec'][r])

    ind = np.array(ind, dtype = 'float64')
    savec = sum(ind)/len(ind)

    # after this line, we use {{max_index}} and {{savec}}
    # 133 ~ 158
    tem = []
    for i in range(1,len(data[cate])): ##ERROR가 난다면 try except -> append(0)
        tem.append(cosine_measure(savec, data[cate][i]))
    
    #160 ~ 165
    title = slice(1,2)
    artist = slice(2,3)
    idx = np.argsort(tem)
    topN = 3
    reco = []
    overlap = set() # 이 변수가 필요한 이유는...같은 노래가 다른 songid로 존재할 경우가 있음.(싱글로 내고 정규에서 또 발매하는경우)
    for i in idx[::-1]:
        if topN==len(reco):
            break
        if math.isnan(tem[i]):
            continue
        if (data[topic][i][title][0], data[topic][i][artist][0]) in overlap:
            continue
        overlap.add((data[topic][i][title][0], data[topic][i][artist][0]))
        reco.append(i)
    rst = {"songs":[]}
    for i in range(topN):
        rst['songs'].append({
            "title": data[topic][reco[i]][title][0],
            "artist":data[topic][reco[i]][artist][0],
            "url":"https://www.youtube.com/results?search_query={0} {1}".format(data[topic][reco[i]][title][0],data[topic][reco[i]][artist][0]) }) 
    return(rst)

if __name__ == '__main__':
    read_data()
    sa = '''
나는 오빠가 너무 좋!아!
오빠가 날 애기취급하는 것도 좋고
나한테 장난치는 것도 좋아!
얼른 우리 만나기로 한 날이 다가왔으면 좋겠어
오빠한테 한번 폭 안기고싶다!! 보고싶어!!
    '''
    print(Djbamboo(sa))