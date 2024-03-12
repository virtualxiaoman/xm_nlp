import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.text import Text
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.chunk import RegexpParser
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn

# nltk.download()

class enWord:
    def __init__(self, word):
        self.word = word
        self.word_wn = None

    def get_meaning(self, meaning_choice=0):
        """
        获取词意
        :param meaning_choice: 选择第几个义项
        """
        self.word_wn = wn.synsets(self.word)
        print("各个词义：", self.word_wn)  # 查看单词的各个词义
        if isinstance(meaning_choice, int) and meaning_choice >= 0:
            print("释义：", self.word_wn[meaning_choice].definition())  # 查看第meaning_choice种词义的解释

    def create(self, meaning_choice=0):
        """
        以词生句
        :param meaning_choice: 选择第几个义项
        """
        if self.word_wn is None:
            self.word_wn = wn.synsets(self.word)
        # 基于第一种词义进行造句
        word_meaning = wn.synsets(self.word)[meaning_choice]  # wn.synsets("dog")[0]相当于wn.synset('dog.n.01')
        print("造句：", word_meaning.examples())
        # 查看这个词的上位词
        print("上位词：", word_meaning.hypernyms())

class enText:
    def __init__(self, text):
        self.raw_text = text
        self.raw_tokens = word_tokenize(self.raw_text)
        self.raw_tokens = [word.lower() for word in self.raw_tokens]
        self.raw_t = Text(self.raw_tokens)
        self.text = text
        self.sentence = None
        self.tokens = None
        self.pos_tag = None
        self.t = None

    def process(self, drop_stopword=True, shallow_process=True, deep_process=False, print_details=True,
                print_stopword_intersection=False):
        """
        预处理
        :param drop_stopword: 是否去除停用词
        :param shallow_process: 是否浅层地预处理(时态变化)
        :param deep_process:  是否深层地预处理
        :param print_details: 是否输出详情(分句和tokens)
        :param print_stopword_intersection: 是否输出输入文本与停用词的交集
        :return:
        """
        self.sentence = nltk.sent_tokenize(self.text)  # 分句子
        # self.text = re.sub('[\u4e00-\u9fa5]', '', self.text)  # 去中文
        word_token = word_tokenize(self.text)  # 分词
        self.tokens = [word.lower() for word in word_token]  # 转小写
        self.tokens = [re.sub(r'[^a-zA-Z]', '', word) for word in self.tokens]  # 去除所有的非字母字符
        if print_stopword_intersection:
            # 是否输出 输入数据 与 英文停用词 的交集
            # print(stopwords.raw('english').replace("\n", " "))  # 查看停用词
            stopword_set = set([word for word in self.tokens])
            print("输入数据与停用词的交集", stopword_set.intersection(set(stopwords.words('english'))))
        if drop_stopword:
            # 是否删除输入数据中的停用词
            self.tokens = [word for word in self.tokens if (word not in stopwords.words('english'))]
        if shallow_process:
            # 语态归一化
            lemmatizer = WordNetLemmatizer()
            self.tokens = [lemmatizer.lemmatize(word) for word in self.tokens]
            # 不建议使用下面两个方法，因为会对人名等错误更改：
            # stemmer = PorterStemmer()
            # self.tokens = [stemmer.stem(word) for word in self.tokens]
            # stemmer1 = SnowballStemmer('english')
            # self.tokens = [stemmer1.stem(word) for word in self.tokens]

        if deep_process:
            # 暂时没学更深入的处理方法
            pass

        if print_details:
            print("分句：", self.sentence)
            print("tokens：", self.tokens)

    def query(self, query_text=None, show_postag=False, show_chunk=False):
        if self.tokens is None:
            self.process()

        self.t = Text(self.tokens)
        self.pos_tag = pos_tag(self.tokens)

        if query_text is not None:
            if isinstance(query_text, list):
                # 查询某些字符串
                for query_text in query_text:
                    print(query_text, "的次数：", self.t.count(query_text))
                    if self.t.count(query_text) > 0:
                        print(query_text, "在原字符串中的位置：", self.raw_t.index(query_text))  # 位置从0开始
                        print(query_text, "在预处理后的字符串中的位置：", self.t.index(query_text))  # 位置从0开始
                    else:
                        print(query_text, "不在输入的文本中")
            elif isinstance(query_text, str):
                # 查询某个字符串
                print(query_text, "的次数：", self.t.count(query_text))
                if self.t.count(query_text) > 0:
                    print(query_text, "在原字符串中的位置：", self.raw_t.index(query_text))  # 位置从0开始
                    print(query_text, "在预处理后的字符串中的位置：", self.t.index(query_text))  # 位置从0开始
                else:
                    print(query_text, "不在输入的文本中")
        if show_postag:
            # 输出post_tag
            print(self.pos_tag)
        if show_chunk:
            # 命名实体识别
            print(ne_chunk(self.pos_tag))

    def plot(self, topN=10, RegexpParser_rule=None):
        if self.tokens is None:
            self.process()
        if self.t is None:
            self.query()

        if isinstance(topN, int) and topN > 0:
            self.t.plot(topN)
        if RegexpParser_rule is not None:
            cp = RegexpParser(RegexpParser_rule)
            result = cp.parse(self.pos_tag)
            result.draw()


class zhText:
    def __init__(self):
        pass


