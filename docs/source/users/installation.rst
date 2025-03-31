# -*- coding: utf-8 -*-
import requests
import json
import time
import pandas as pd
import jieba
import re
from bs4 import BeautifulSoup
import os
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 设置日志
logging.basicConfig(level=logging.INFO)

# ------------------------ 数据爬取 ------------------------
def get_comments(song_id=254574, max_pages=5, delay=1.5):
    """获取网易云音乐评论数据"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'Referer': 'https://music.163.com/'
    }
    
    base_url = f"https://music.163.com/api/v1/resource/comments/R_SO_4_{song_id}"
    comments = []
    
    for page in range(max_pages):
        try:
            params = {
                'rid': f'R_SO_4_{song_id}',
                'offset': page * 20,
                'limit': 20,
                'csrf_token': ''
            }
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            response.encoding = 'utf-8'
            data = json.loads(response.text)
            
            for comment in data.get('comments', []):
                clean_content = BeautifulSoup(comment['content'], 'html.parser').get_text()
                comments.append({
                    'content': clean_content,
                    'liked_count': comment['likedCount'],
                    'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(comment['time']/1000))
                })
            
            print(f"已获取第{page+1}页，累计{len(comments)}条评论")
            time.sleep(delay)
            
            if not data.get('more', False):
                break
                
        except Exception as e:
            logging.error(f"第{page+1}页获取失败: {str(e)}")
    
    return pd.DataFrame(comments)

# ------------------------ 数据预处理 ------------------------
def preprocess_text(df, stopwords_path='chinese_stopwords.txt'):
    """数据清洗与分词"""
    # 去除空值
    df = df[df['content'].str.strip() != ''].copy()
    
    # 加载停用词
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    
    # 分词并去停用词
    df['tokens'] = df['content'].apply(
        lambda x: [word for word in jieba.lcut(x) 
                   if word not in stopwords and len(word) > 1 and re.match('^[\u4e00-\u9fa5]+$', word)]
    )
    
    return df

# ------------------------ 情感分析 ------------------------
def load_sentiment_dict(path='bosonnlp_sentiment_score.txt'):
    """加载情感词典"""
    sentiment_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                word, score = line.strip().split('\t')
                sentiment_dict[word] = float(score)
    return sentiment_dict

def calculate_sentiment(tokens, sentiment_dict):
    """计算情感得分"""
    score = sum(sentiment_dict.get(word, 0) for word in tokens)
    return score

# ------------------------ 模型训练 ------------------------
def train_model(df, sentiment_dict):
    """构建SVM模型"""
    # 生成情感标签
    df['sentiment_score'] = df['tokens'].apply(lambda x: calculate_sentiment(x, sentiment_dict))
    df['label'] = df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0.5 else 'negative' if x < -0.3 else 'neutral'
    )
    
    # TF-IDF特征
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df['tokens'].apply(lambda x: ' '.join(x)))
    
    # 情感特征
    sentiment_features = np.array([
        [sum(1 for word in tokens if sentiment_dict.get(word, 0) > 0) / len(tokens) if len(tokens) > 0 else 0,
         sum(1 for word in tokens if sentiment_dict.get(word, 0) < 0) / len(tokens) if len(tokens) > 0 else 0,
         abs(score)]
        for tokens, score in zip(df['tokens'], df['sentiment_score'])
    ])
    
    # 合并特征
    from scipy.sparse import hstack
    X = hstack([X_tfidf, sentiment_features])
    y = df['label']
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 训练SVM
    svm = SVC(kernel='linear', class_weight='balanced', C=1.0)
    svm.fit(X_train, y_train)
    
    # 评估模型
    y_pred = svm.predict(X_test)
    print("准确率:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return svm, tfidf

# ------------------------ 可视化 ------------------------
def plot_sentiment_distribution(df):
    """绘制情感分布饼图"""
    counts = df['label'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('网易云音乐评论情感分布')
    plt.show()

def generate_wordcloud(df):
    """生成词云"""
    text = ' '.join(df['tokens'].apply(lambda x: ' '.join(x)))
    wordcloud = WordCloud(
        font_path='SimHei.ttf', 
        background_color='white',
        width=800, 
        height=600
    ).generate(text)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('评论关键词词云')
    plt.show()

# ------------------------ 主程序 ------------------------
if __name__ == "__main__":
    # 爬取数据
    df = get_comments(song_id=254574, max_pages=5)  # 示例歌曲《山雀》
    df.to_csv('music_comments.csv', index=False, encoding='utf_8_sig')
    
    # 数据预处理
    df = preprocess_text(df)
    
    # 情感分析
    sentiment_dict = load_sentiment_dict()
    df['sentiment_score'] = df['tokens'].apply(lambda x: calculate_sentiment(x, sentiment_dict))
    df['label'] = df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0.5 else 'negative' if x < -0.3 else 'neutral'
    )
    
    # 模型训练
    svm, tfidf = train_model(df, sentiment_dict)
    
    # 可视化
    plot_sentiment_distribution(df)
    generate_wordcloud(df)
