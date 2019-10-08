# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 01:48:02 2019

@author: Jean-Philippe
"""

import random as ran
#Basic 1------------------------------
def unduplicated(alist):
    Nodup=[]
    for I in alist:
        if I not in Nodup:
            Nodup.append(I)
   
    print(Nodup)
    return Nodup

#test
L=[1,1,1,2,2,2,3,3,3]
LND=unduplicated(L)


#Basic 2-------------------------------
def isinlist(alist,aelement):
    print(aelement in alist)

#test
L=[1,3,5,7,9]
X=isinlist(L,3)


#Basic 3-------------------------------
a=[1,4,9,16,25,36,49,64,81,100]
aeven=[n for n in a if n%2==0]

#test
print(aeven)

#Advanced 1----------------------------
def guideuser(r,bet):
    dif=bet-r
    if dif>50: print('Waaaay Lower')
    if 5<=dif<=50: print('Lower')    
    if dif<-50: print('Waaaay Higher')  
    if -5>=dif>=-50: print('Higer') 
    if 5>dif>-5: print('Close')
    if dif==0: print('BINGO')
    
    
def guideuser2(r,bet):  
    s=''
    if r%2==0: 
        s=s+'No! It is a even number'
    else:
        s=s+'No! IT is a odd number'

    if r%3==0: 
        s=s+'and it is divisible by 3'
    else:
        s=s+'and it is not divisible by 3'
    print(s)

def playGuess1():
     
    r=ran.randrange(0,201)
    #methode 1
    bet=int(input('Guess between 0 to 200: '))
    if r==bet: 
        print('BINGO \n')
    else:    
        while r!=bet:
            guideuser(r,bet)
            bet=int(input('Guess between 0 to 200: '))
        print('BINGO \n') 

def playGuess2():
    pickm1=1  
    pickm2=1
    npick=[]
    r=ran.randrange(0,201)
    #methode 1
    bet=int(input('Guess between 0 to 200: '))
    if r==bet: 
        print('BINGO \n')
    else:    
        while r!=bet:
            guideuser(r,bet)
            bet=int(input('Guess between 0 to 200: '))
            pickm1=pickm1+1  
        print('BINGO /n') 
    npick.append(pickm1)
    #methode 2
    bet=int(input('Guess between 0 to 200: '))
    if r==bet: 
        print('BINGO \n')
    else:    
        while r!=bet:
            if pickm2>2:
                guideuser2(r,bet)
            else:
                guideuser(r,bet)
            bet=int(input('Guess between 0 to 200: '))
            pickm2=pickm2+1  
        print('BINGO \n')    
    npick.append(pickm2)
    return(npick)
    
#Test1    
#print("Play Guess a number")
#playGuess1()

#Test3
#print("Play Guess 2 number")
#result=playGuess2()
#print("Bravo you got it in "+ str(result[0]) + " try and "+ str(result[1]) + " try.\n")


#Advanced 2----------------------------
def askguess(rep,lastguess,tolow,tohigh,close):
    maxmin=0
    minmax=200
    
    if len(tolow)>=1:
        maxmin=max(tolow)
    if len(tohigh)>=1:
        minmax=min(tohigh)
    if len(close)>=1:    
        maxminc=max(close)-10
        minmaxc=min(close)+10
        minmax=min(minmax,minmaxc)    
        maxmin=max(maxmin,maxminc)   
        
    if rep==1:
        newguess=ran.randrange(max(0,maxmin),min(lastguess-50,minmax))
    if rep==2:
        newguess=ran.randrange(max(lastguess-50,maxmin),min(lastguess-5,minmax))    
    if rep==3:
        newguess=ran.randrange(max(lastguess+50,maxmin),min(201,minmax))
    if rep==4:
        newguess=ran.randrange(max(lastguess+5,maxmin),min(lastguess+50,minmax))  
    if rep==5:
        r=ran.randrange(max(lastguess-5,maxmin),min(lastguess+5,minmax)) 
        while r in close:
            r=ran.randrange(max(lastguess-5,maxmin),min(lastguess+5,minmax)) 
        newguess=r
    if rep==6:
        newguess=-1
    return newguess

def askguess2(rep,lastguess,tolow,tohigh,close,even,divby3):
    maxmin=0
    minmax=200
    x=0
    if len(tolow)>=1:
        maxmin=max(tolow)
    if len(tohigh)>=1:
        minmax=min(tohigh)
    if len(close)>=1:    
        maxminc=max(close)-10
        minmaxc=min(close)+10
        minmax=min(minmax,minmaxc)    
        maxmin=max(maxmin,maxminc)   
    r=ran.randrange(maxmin,minmax)
    e=False
    d=False
    print(even)
    print(divby3)
    if r%2==0:
        e=True
    if r%3==0:
        d=True  
    while r in close or e!=even or divby3!=d:
        r=ran.randrange(maxmin,minmax)
        x=x+1
        if r%2==0:
            e=True
        if r%3==0:
            d=True 
        if x>200:
            break
    newguess=r
    return newguess   

def PlayMakeCPUGuess():    
    print('Picks a number between 0 and 200. Then CPU I will try to guess and you should guide it')
    if str(input('Type y if you are ready : '))=='y':
        r=ran.randrange(0,201)
        print('My guess is : ' + str(r))
        lastguess=r
        rep=int(input('Tape 1 for: Way Lower, 2 for: Lower, 3 for: Way Higher, 4 for: Higher, 5 for: Close, 6 for: BINGO : '))
        tolow=[]
        tohigh=[]
        close=[]
        if rep==1: tohigh.append(r-50)
        if rep==2:tohigh.append(r-5)
        if rep==3:tolow.append(r+50)
        if rep==4:tolow.append(r+5)
        if rep==5:close.append(r)
        
        while rep!=6:
            r=askguess(rep,lastguess,tolow,tohigh,close)
            print('My guess is : ' + str(r))
            lastguess=r
            rep=int(input('Tape 1 for: Way Lower, 2 for: Lower, 3 for: Way Higher, 4 for: Higher, 5 for: Close, 6 for: BINGO : '))
            if rep==1: tohigh.append(r-50)
            if rep==2:tohigh.append(r-5)
            if rep==3:tolow.append(r+50)
            if rep==4:tolow.append(r+5)
            if rep==5:close.append(r)
        print('Yayyyyyyyyyyy')
        

def PlayMakeCPUGuess2():    
    print('Picks a number between 0 and 200. Then CPU I will try to guess and you should guide it')
    if str(input('Type y if you are ready : '))=='y':
        pickm1=1  
        
        npick=[]
        r=ran.randrange(0,201)
        print('My guess is : ' + str(r))
        lastguess=r
        rep=int(input('Tape 1 for: Way Lower, 2 for: Lower, 3 for: Way Higher, 4 for: Higher, 5 for: Close, 6 for: BINGO : '))
        tolow=[]
        tohigh=[]
        close=[]
        if rep==1: tohigh.append(r-50)
        if rep==2:tohigh.append(r-5)
        if rep==3:tolow.append(r+50)
        if rep==4:tolow.append(r+5)
        if rep==5:close.append(r)
        #method1y
        while rep!=6:
            r=askguess(rep,lastguess,tolow,tohigh,close)
            print('My guess is : ' + str(r))
            lastguess=r
            rep=int(input('Tape 1 for: Way Lower, 2 for: Lower, 3 for: Way Higher, 4 for: Higher, 5 for: Close, 6 for: BINGO : '))
            if rep==1: tohigh.append(r-50)
            if rep==2:tohigh.append(r-5)
            if rep==3:tolow.append(r+50)
            if rep==4:tolow.append(r+5)
            if rep==5:close.append(r)
            pickm1=pickm1+1
        print('Yayyyyyyyyyyy')
        npick.append(pickm1)
        #method2
        print('Picks a number between 0 and 200. Then CPU I will try to guess and you should guide it')
        if str(input('Type y if you are ready : '))=='y':
            pickm2=1
            r=ran.randrange(0,201)
            print('My guess is : ' + str(r))
            lastguess=r
            rep=int(input('Tape 1 for: Way Lower, 2 for: Lower, 3 for: Way Higher, 4 for: Higher, 5 for: Close, 6 for: BINGO : '))
            tolow=[]
            tohigh=[]
            close=[]
            if rep==1: tohigh.append(r-50)
            if rep==2:tohigh.append(r-5)
            if rep==3:tolow.append(r+50)
            if rep==4:tolow.append(r+5)
            if rep==5:close.append(r)
            while rep!=6 :
                if pickm2<=2:
                    r=askguess(rep,lastguess,tolow,tohigh,close)
                    print('My guess is : ' + str(r))
                    lastguess=r
                    rep=int(input('Tape 1 for: Way Lower, 2 for: Lower, 3 for: Way Higher, 4 for: Higher, 5 for: Close, 6 for: BINGO : '))
                    if rep==1: tohigh.append(r-50)
                    if rep==2:tohigh.append(r-5)
                    if rep==3:tolow.append(r+50)
                    if rep==4:tolow.append(r+5)
                    if rep==5:close.append(r)
                    pickm2=pickm2+1
                else:

                    if pickm2==3:
                        even=False
                        divby3=False
                        r=askguess(rep,lastguess,tolow,tohigh,close)
                        print('My guess is : ' + str(r))
                        lastguess=r
                    if pickm2>3:
                         r=askguess2(rep,lastguess,tolow,tohigh,close,even,divby3)
                         print('My guess is : ' + str(r))
                         lastguess=r
                         
                    
                    rep=int(input('Tape 1 if it is a even number: 2 for odd :  6 for: BINGO : '))
                    if rep==1:
                        even=True
                    if rep!=6:
                        rep=int(input('Tape 1 if it is a number divisible by 3: 2 If not :  6 for: BINGO : '))
                        if rep==1:
                            divby3=True
                    pickm2=pickm2+1
        print('Yayyyyyyyyyyy')
        npick.append(pickm2)
        return npick
#Test2
#PlayMakeCPUGuess()        

#test4        
n=[]    
n=PlayMakeCPUGuess2()     
print("Good job CPU got it in "+ str(n[0]) + " try and "+ str(n[1]) + " try.\n")    
    
    
    
    
    
    
    
    
    
    