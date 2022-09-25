import sys
import math
import string

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    #create an alphabatic dict with each value 0
    X=dict.fromkeys(string.ascii_uppercase, 0)
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        for line in f:
            for letter in line:
                upper_l = letter.upper()
                if upper_l in string.ascii_uppercase and upper_l in X:
                    X[upper_l] += 1
    return X


# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

#print num (round and 4 decimal places)
def prt_round(num):
    print(format(round(num, 4), '.4f'))

#X = shred("samples/letter0.txt")
#X = shred("samples/letter1.txt")
#X = shred("samples/letter2.txt")
#X = shred("samples/letter3.txt")
X = shred("letter.txt")

#Q1
print("Q1")
for k,v in X.items():
    print(k,v)

#Q2
e,s = get_parameter_vectors()
print("Q2")
X1 = X["A"]
loge1 = math.log(e[0])
logs1 = math.log(s[0])
prt_round(X1*loge1)
prt_round(X1*logs1)

#Q3
print("Q3")
engPrior = 0.6
spaPrior = 0.4
logYEng = math.log(engPrior)
logYSpa = math.log(spaPrior)
X_l = list(X.values())
def sumXp(p):
    sum = 0
    for i in range(26):
        sum += X_l[i]*math.log(p[i])
    return sum
sumEng = sumXp(e)
sumSpa = sumXp(s)
FEng = logYEng + sumEng
FSpa = logYSpa + sumSpa
prt_round(FEng)
prt_round(FSpa)

#Q4
print("Q4")
diffSpaEng = FSpa - FEng
if diffSpaEng >= 100: prt_round(0)
if diffSpaEng <= -100: prt_round(1)
result_q4 = 1/(1+math.exp(diffSpaEng))
prt_round(result_q4)