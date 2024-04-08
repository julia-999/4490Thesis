# import libraries
import torch
import torchtext
import gensim.downloader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
import warnings 
warnings.filterwarnings("ignore")

# import the pre-trained models
fasttext = torchtext.vocab.FastText(language="en")
glove = torchtext.vocab.GloVe(name="42B")
word2vec = gensim.downloader.load('word2vec-google-news-300')
excel_file = "./Mental Health Word Associations(1-21).xlsx"
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(["apathy", "tiredness", "sleeplessness", "helplessness"])
model.resize_token_embeddings(len(tokenizer))

matrix_dict = dict()
matrix_words = []
errors = []


def in_vocab(target_words, algorithm):
    
    if algorithm == "glove":

        if target_words[0] not in glove.stoi:
            print("not in glove vocab: " + target_words[0])
            return False
        if target_words[1] not in glove.stoi:
            print("not in glove vocab: " + target_words[1])
            return False
    
    if algorithm == "fasttext":

        if target_words[0] not in fasttext.stoi:
            print("not in fasttext vocab: " + target_words[0])
            return False
        if target_words[1] not in fasttext.stoi:
            print("not in fasttext vocab: " + target_words[1])
            return False
    
    if algorithm == "word2vec":

        if target_words[0] not in word2vec.key_to_index:
            print("not in word2vec vocab: " + target_words[0])
            return False
        if target_words[1] not in word2vec.key_to_index:
            print("not in word2vec vocab: " + target_words[1])
            return False

    return True


def cosine_similarity(target_words, algorithm):

    if algorithm == "glove":
        cosine_glove = torch.cosine_similarity(glove[target_words[0]].unsqueeze(0), glove[target_words[1]].unsqueeze(0))
        return float(cosine_glove)
    
    if algorithm == "fasttext":
        cosine_fasttext = torch.cosine_similarity(fasttext[target_words[0]].unsqueeze(0), fasttext[target_words[1]].unsqueeze(0))
        return float(cosine_fasttext)
    
    if algorithm == "word2vec":
        cosine_word2vec = np.dot(word2vec[target_words[0]], word2vec[target_words[1]])/(np.linalg.norm(word2vec[target_words[0]])* np.linalg.norm(word2vec[target_words[1]]))
        return float(cosine_word2vec)

    return 0


def dot_product(target_words, algorithm):
    
    if algorithm == "glove":
        dot_glove = torch.dot(glove[target_words[0]], glove[target_words[1]])
        return float(dot_glove)
    
    if algorithm == "fasttext":
        dot_fasttext = torch.dot(fasttext[target_words[0]], fasttext[target_words[1]])
        return float(dot_fasttext)

    if algorithm == "word2vec":
        dot_word2vec = np.dot(word2vec[target_words[0]], word2vec[target_words[1]])
        return float(dot_word2vec)
    
    return 0


def process_excel(file):

    exc = pd.read_excel(file)
    col = [col for col in exc.columns if col.endswith('2')]
    wm = exc.loc[:, col]
    wm = wm.transpose()
    return wm


def construct_excel_matrix(target, df):

    max = 4*len(df.columns)
    dfnew = pd.DataFrame(max, index=target, columns=target)
    
    for i in df.iterrows():
        r = i[0].lower().strip("2")
        for j in range(1, len(i)):
            wrds = (i[j].str.split(" "))
            for k in wrds:
                k = k[0].split(";")
                for index in range(len(k)-1):
                    dfnew.loc[r, k[index].lower().strip("2")] -= index

    return dfnew


def process_excel_contextual(file):

    d = dict()
    exc = pd.read_excel(file)

    for col in exc.columns:
        if col in ["Joy", "Sadness", "Melancholy", "Apathy", "Anxiety", "Happy", "Neutral", "Sad", "Love", "Relaxed", "Hate", "Uneasy", "Suffering", "Depression", "Nightmares", "Angry", "Pleasure", "Tiredness", "Fine", "Motivation", "Stress", "Sleeplessness", "Pain", "Anger", "Helplessness"]:
            d[col] = []
            for index, row in exc.iterrows():
                if not pd.isnull(row[col]):
                    d[col].append(row[col])
    return d

# reference: https://github.com/arushiprakash/MachineLearning/blob/main/BERT%20Word%20Embeddings.ipynb
def prepare_bert(text, tokenizer):

    tokenized = tokenizer.tokenize("[CLS] " + text + " [SEP]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
    tokens = torch.tensor([indexed_tokens])
    segments = torch.tensor([[1]*len(indexed_tokens)])

    return tokenized, tokens, segments

# reference: https://github.com/arushiprakash/MachineLearning/blob/main/BERT%20Word%20Embeddings.ipynb
def get_embeddings_bert(tokens, segments, model):
    
    with torch.no_grad():
        outputs = model(tokens, segments)
        hidden_states = outputs[2][1:]

    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)

    return [token_embed.tolist() for token_embed in token_embeddings]


def compare_matrices(m1, m2):

    m1 = m1.fillna(0)
    m2 = m2.fillna(0)

    m1 = (m1 - m1.min()) / (m1.max() - m1.min())
    m2 = (m2 - m2.min()) / (m2.max() - m2.min())

    result = m1.subtract(m2)
    return np.sum(result.values)


def contextual():

    wm = process_excel(excel_file)

    with open('wordpairs.txt') as file, open('juliaExperiment1.md', 'w') as file2:
       
        line = file.readline()
        
        while line != "word_pairs\n":
            line = file.readline()
        
        line = file.readline()

        while line != "\n":
            
            target_words = [word.strip().lower() for word in line.split(',')]

            # loop through algorithms
            for a in ["glove", "fasttext", "word2vec"]:

                # for storing results
                cosinelist = []
                dotlist = []

                # create the different combinations of words and resets iterator
                permutations = itertools.product(target_words, target_words)

                for i in permutations:
                    if in_vocab(i, a):
                        cosinelist.append([i, cosine_similarity(i, a)])
                        dotlist.append([i, dot_product(i, a)])
                    else:
                        raise ValueError("Not in " + a + " vocabulary list: " + str(i))


                # format matrix
                m = []
                m.append([" "] + target_words)

                m2 = []
                m2.append([" "] + target_words)

                for i in range(0, 5):
                    m.append([target_words[i], cosinelist[i*5][1], cosinelist[i*5+1][1], cosinelist[i*5+2][1], cosinelist[i*5+3][1], cosinelist[i*5+4][1]])
                    m2.append([target_words[i], dotlist[i*5][1], dotlist[i*5+1][1], dotlist[i*5+2][1], dotlist[i*5+3][1], dotlist[i*5+4][1]])

                df = pd.DataFrame(m[1:], columns=m[0])
                df2 = pd.DataFrame(m2[1:], columns=m2[0])
                df = df.set_index([" "])
                df2 = df2.set_index([" "])

                file2.write("\n# " + a + " Cosine Similarity Matrix for "+ line + "  \n")
                df.to_markdown(buf=file2)
                file2.write("\n# " + a + " Dot Product Matrix for "+ line + "  \n")
                df2.to_markdown(buf=file2)

                fline = "_".join(line.split(", ")).strip("\n")

                for i in [df, df2]:
                    for j in range(5):
                        i.iat[j, j] = None

                plt.imshow(df, cmap="plasma_r")
                plt.title(a + " Cosine Similarity Matrix for "+ line)
                plt.xticks(np.arange(5), target_words)
                plt.yticks(np.arange(5), target_words)
                plt.savefig("./colormaps/c_" + a + "_" + fline + ".png")
                matrix_dict[fline + "_" + a + "_c"] = df

                plt.imshow(df2, cmap="plasma_r")
                plt.title(a + " Dot Product Matrix for "+ line)
                plt.xticks(np.arange(5), target_words)
                plt.yticks(np.arange(5), target_words)
                plt.savefig("./colormaps/d_" + a + "_" + fline + ".png")
                matrix_dict[fline + "_" + a + "_d"] = df2

            # input analysis
            filtered = wm[wm.index.str.lower().str[:-1].isin(tuple(target_words))]
            dfi = construct_excel_matrix(target_words, filtered)

            file2.write("\n# Input Ranking Matrix for "+ line + "  \n")
            dfi.to_markdown(buf=file2)

            for j in range(5):
                dfi.iat[j, j] = None

            plt.imshow(dfi, cmap="plasma_r")
            plt.title(a + " Input Ranking Matrix for "+ line + "")
            plt.xticks(np.arange(5), target_words)
            plt.yticks(np.arange(5), target_words)
            plt.savefig("./colormaps/i_" + fline + ".png")
            matrix_dict[fline + "_i"] = dfi

            line = file.readline()


def noncontextual():

    # dictionary of inputs
    di = process_excel_contextual(excel_file)
    dp = dict()
    davg = dict()

    list_of_distances = []
    list_of_avg_distances = []
    all_sentences = []

    for key, value in di.items():
        dp[key] = dict()
        temp = []
        for v in value:
            tokenized, tokens, segments = prepare_bert(v, tokenizer)
            list_token_embeddings = get_embeddings_bert(tokens, segments, model)
            try:
                word_index = word_index = tokenized.index(key.lower())
                all_sentences.append(v)
                word_embedding = list_token_embeddings[word_index]
                dp[key][v] = word_embedding
                temp.append(word_embedding)
            except Exception as e:
                # print("not in list")
                errors.append(repr(e) + " ---- " + v + " tokenized: " + repr(tokenized))

        # for average
        davg[key] = np.mean(temp, axis=0)

    for key1, value1 in dp.items():
        for i, j in value1.items():
            for key2, value2 in dp.items():
                for i2, j2 in value2.items():
                    cos_dist = 1 - cosine(j, j2)
                    list_of_distances.append([i, i2, cos_dist])

    words = []
    for key1, value1 in davg.items():
        words.append(key1)
        for key2, value2 in davg.items():
            try:
                cos_dist = 1 - cosine(value1, value2)
                list_of_avg_distances.append([key1, key2, cos_dist])
            except: 
                errors.append("Issue with cos_dist: " + key1 + " | " + key2)

    distance_matrix = pd.DataFrame(0, index=all_sentences, columns=all_sentences)
    avg_distance_matrix = pd.DataFrame(0, index=words, columns=words)

    for distance in list_of_distances:
        i, i2, cos_dist = distance
        distance_matrix.at[i, i2] = cos_dist

    for distance in list_of_avg_distances:
        i, i2, cos_dist = distance
        avg_distance_matrix.at[i, i2] = cos_dist

    plt.imshow(distance_matrix, cmap="plasma_r")
    plt.savefig("./distances/allcombos.png")

    plt.imshow(avg_distance_matrix, cmap="plasma_r")
    plt.xticks(np.arange(len(words)), words)
    plt.yticks(np.arange(len(words)), words)
    plt.savefig("./distances/allcombos.png")

    with open('wordpairs.txt') as file:
       
        line = file.readline()
        
        while line != "word_pairs\n":
            line = file.readline()
        
        line = file.readline()

        while line != "\n":
            
            target_words = [word.strip().lower() for word in line.split(',')]
            matrix_words.append(target_words)

            small_matrix = pd.DataFrame(0, index=target_words, columns=target_words)
            for i in target_words:
                for j in target_words:
                    small_matrix.at[i, j] = float(avg_distance_matrix.at[i.capitalize(), j.capitalize()])

            fline = "_".join(line.split(", ")).strip("\n")

            diagonal = pd.DataFrame(None, index=target_words, columns=target_words, dtype=float)
            for j in range(5):
                small_matrix.iat[j, j] = None
                diagonal.iat[j, j] = 0

            plt.imshow(small_matrix, cmap="plasma_r", interpolation=None)
            plt.imshow(diagonal, cmap='gray_r')
            plt.xticks(np.arange(len(target_words)), target_words)
            plt.yticks(np.arange(len(target_words)), target_words)
            plt.savefig("./colormaps/t_" + fline + ".png")
            matrix_dict[fline + "_t"] = small_matrix
        
            line = file.readline()    


def main():
    contextual()
    noncontextual()

    scores = []

    for i, i2 in matrix_dict.items():
        for j, j2 in matrix_dict.items():
            if i.split("_")[0] == j.split("_")[0] and i != j:
                score = abs(compare_matrices(i2, j2))
                scores.append([i, j, score])

    for w in matrix_words:
        labels = [label for label in matrix_dict.keys() if label.startswith(w[0])]
        score_df = pd.DataFrame(0, index=labels, columns=labels)
        for s in scores:
            if s[0].startswith(w[0]):
                i, i2, score = s
                score_df.at[i, i2] = score

        mtype = ["_".join(n.split("_")[5:]) for n in score_df.index]

        mtype = ["GloVe \nCosine Similarity", "GloVe \nDot Product", "fastText \nCosine Similarity", "fastText \nDot Product", "Word2vec \nCosine Similarity", "Word2vec \nDot Product", "Participant \nRankings", "Participant Sentences + \nBERT Cosine Similarity"]

        plt.imshow(score_df, cmap="plasma", interpolation=None)
        plt.title("Method Similarity Comparison for \n" + " ".join(w))
        plt.xticks(np.arange(len(mtype)), mtype, rotation=90)
        plt.yticks(np.arange(len(mtype)), mtype)
        plt.savefig("./distances/" + w[0] + "scores.png", bbox_inches="tight")

    print("Summary of errors: ")
    for i in errors:
        print(i)


if __name__ == "__main__":
    main()
