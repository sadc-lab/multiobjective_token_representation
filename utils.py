import numpy as np
from sklearn.metrics import confusion_matrix

def stats(data):
    average = np.mean(data, axis=0)
    std =  np.mean((data-average)**2, axis=0)
    std = np.sqrt(std)
    return average, std

def decode_properly(tokenizer_output, tokenizer):
    """
    decode batch of tokenized sentences into sentences
    """
    decoded_tokenized_sents = [sent.replace('<pad>', '').replace('<mask>', ' <mask>') for sent in tokenizer.batch_decode(tokenizer_output.input_ids)]
    return decoded_tokenized_sents

def custom_precision_recall_score(y_true, y_pred, nb_labels):
    matrix = confusion_matrix(y_true, y_pred, labels = list(range(nb_labels)))
    precision = np.zeros(nb_labels)
    recall = np.zeros(nb_labels)
    sum_cols = np.sum(matrix, axis = 0)
    sum_lines = np.sum(matrix, axis = 1)
    
    for i in range(nb_labels):
        sum_col_i = sum_cols[i] + 1e-7
        sum_line_i = sum_lines[i] + 1e-7
        precision[i] = matrix[i, i]/sum_col_i
        recall[i] = matrix[i, i]/sum_line_i
        
    return precision, recall

def custom_f1_score(y_true, y_pred, nb_labels, average=None):
    """
    average = None or "macro"
    """
    precision, recall = custom_precision_recall_score(y_true, y_pred, nb_labels)
    scores = 2*precision*recall/(precision + recall + 1e-7)
    
    return scores if average == None else scores.mean()
    
def remove_punctuation(word):
    """
    This function cleans useless ponctuation from text. It is meant to be called 
    on each word returned by sentence.split()
    """
    if len(word) == 0:
        return []
    if len(word) == 1:
        return [word]
    
    if word[0] in [':', ',', '.']:
        return [word[0]] + remove_punctuation(word[1:])
    
    idx = 0
    while (idx < len(word)) and (not word[idx] in ['!', '(', ')', ';']) :
        idx += 1
    
    if idx == 0:
        return [word[0]] + remove_punctuation(word[1:])
    if idx == len(word):
        if word[-1] in ['.', ',', ':']:
            return [word[:-1], word[-1]]
        else:
            return [word]
    
    return [word[:idx], word[idx]] + remove_punctuation(word[idx+1:])


def intermediate(word):
    """
    This functions aims at isolating numbers. It is meant to be called on each word after
    the remove_punctuation function
    """
    
    # deals with apostrophes at the end of numbers
    result = re.sub(r"(\d)'$", r"\1", word)
    #convert scientific notation to arabic notation
    result = re.sub(r"(^|\s)(\d+e\d+|\d+E\d+|\d+e-\d+|\d+E-\d+)", lambda x: x.group(1)+str(float(x.group(2))), result)
    #creates a space separation for percentages
    result = re.sub(r"(\d)%", r"\1 %", result)
    #separates equalites (FR=50)
    result = re.sub(r"=([^\s])", r" = \1", result)
    # deals with certain codes 2q11.5q12-->2q11.5 - 2q12
    result = re.sub(r"(^|\s)(\d+)(q|p)(\d+[\.,]\d+|\d+)-?(p|q|/)(\d+[\.,]\d+|\d+)", r"\1\2\3\4 - \2\3\6", result)
    result = re.sub(r"(^|\s)(\d+)(q|p)(\d+[\.,]\d+|\d+)-?(p|q|/)(\d+[\.,]\d+|\d+)", r"\1\2\3\4 - \2\3\6", result)
    result = re.sub(r"(^|\s)(\d+)(q|p)(\d+[\.,]\d+|\d+)-?(p|q|/)(\d+[\.,]\d+|\d+)", r"\1\2\3\4 - \2\3\6", result)
    result = re.sub(r"(^|\s)(\d+)\.(q|p)\.(\d+[\.,]\d+|\d+)", r"\1\2\3\4", result)
    
    #separates range of numbers (except apgar) and operations (32+6)
    result = re.sub(r"(\d|%)(-+|\++|/+|:+|x+|X+|\|+|>+|<+|\*|~)(\d)", r"\1 \2 \3", result)
    #runs again in case there is apgar or a date
    result = re.sub(r"(\d|%)(-+|\++|/+|:+|x+|X+|\|+|>+|<+|\*|~)(\d)", r"\1 \2 \3", result)

    # deals with certain numbers representation 2kg48-->2.48kg
    result = re.sub(r"(^|\s)(\d+)(kg|KG|Kg)(\d+)", r"\1\2.\4\3", result)
    # deals with time representation 2h48-->2 h 48
    result = re.sub(r"(^|\s)(\d+)(h|H)(\d+)", r"\1\2 \3 \4", result)
    
    #specifically targets apgar (8.8.8) and dates
    result = re.sub(r"^(\d+)(\.|,)(\d+)(\.|,)(\d+)", r"\1 - \3 - \5", result)
    #deals with lists
    result = re.sub(r"(\s|^)(\d+[\.)])([A-Za-zÀ-ÿ]{2,})(\s|$)", r"\1item) \3\4", result)
    
    # remove noise at the end of numbers (12-->)
    result = re.sub(r"(^|\s)(\d+[\.,]\d+)\.([^A-Za-z0-9\s]+)", r"\1\2 \3", result)
    result = re.sub(r"(^|\s)(\d+[\.,]\d+)([^A-Za-z0-9\s\.]+)", r"\1\2 \3", result)
    if not re.search(r"(^|\s)(\d+[\.,]\d+)", result): #when the format d,d is not detected
        result = re.sub(r"(^|\s)(\d+)\.([^A-Za-z0-9\s]+)", r"\1\2 \3", result)
        result = re.sub(r"(^|\s)(\d+)([^A-Za-z0-9\s\.]+)", r"\1\2 \3", result)
        
    result = re.sub(r"(\d)(:|/)([A-Za-z])", r"\1 \2 \3", result)
    result = re.sub(r"(\d),([A-Za-z])", r"\1 , \2", result)
    result = re.sub(r"(\d)\.([A-Za-pr-z])", r"\1 \2", result)
    # remove noise at the beginning of numbers (-->12)
    result = re.sub(r"(\s|^)([^A-Za-z0-9\s]+)(\d+[\.,]\d+|\d+)", r"\1\2 \3", result)
    result = re.sub(r"([A-Za-z]+),(\d)", r"\1, \2", result)
    result = re.sub(r"([A-Za-pr-z]+)\.(\d)", r"\1 \2", result)
    result = re.sub(r"([A-Za-pr-z]+)(\d+[\.,]\d+)", r"\1 \2", result)
    result = re.sub(r"(^|\s)(TA|TVC|FC|Fc|fc|min|q|Q)(\d)", r"\1\2 \3", result) 
    
    #separates numbers and units
    result = re.sub(r"(\s|^)(\d+[\.,]\d+|\d+)([A-Za-pr-z])", r"\1\2 \3", result)
    # deals with operations within texts (x12 or SA+2jr)
    result = re.sub(r"(x|\+|X|:|<|>|/|-|«)(\d)", r" \1 \2", result)
    result = re.sub(r"([A-Za-z0-9])->", r"\1 ->", result)
    result = re.sub(r"->([A-Za-z0-9])", r"-> \1", result)
    
    # deals with LetterDigit codes (J2, ccq3h)
    result = re.sub(r"(^|\s)([JGSjgs])(\d+)(\s|$)", r" \1\2 \3\4", result)
    result = re.sub(r"ccq(\d+)", r" cc q \1", result)
    # remove noise at the beginning of words (S02->SO2)
    result = re.sub(r"(^|\s)(-|\.|,|\+)([A-Za-z])", r"\1\2 \3", result)
    # remove noise at the beginning of words (-SO2)
    result = re.sub(r"([^0-9])02(\s|$)", r"\1O2\2", result)
    # remove noise at the end of words (SO2-)
    result = re.sub(r"([^\s])-(\s|$)", r"\1 - \2", result)

    return result

def minor_edit(word):
    """
    we run the intermediate function twice to ensure all the issues are properly solved
    """
    edited_word = intermediate(word).split()
    result = []
    for token in edited_word:
        result.append(intermediate(token))
    return ' '.join(result)

def number_masking(word):
    """
    Returns the blinded word
    """
    #single number
    if re.search(r"^(\d+[,\.]\d+|\d+\.?)$", word):
        #1 is very often used as a determinant than a number
        if word == "1":
            return word
        return "nombre"
    if re.search(r"^(\d+[,\.]\d+|\d+)\.$", word):
        return "nombre"

    return word
    