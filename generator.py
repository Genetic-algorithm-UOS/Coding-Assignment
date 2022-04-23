import pandas as pd
import random
import os  
import copy
import math

class DataGenerator:
    
    '''
        This class serves to create a dataset in the form of a pd.dataframe. 
        This dataframe can either include basic features or all available 
        features depending on which internal method is called.  
    '''

    def __init__ (self):
        
        '''
            Prompts user input with regards to the desired sample size. Said input needs to be
            of type integer, dividable by 10 and of size 10 - 2700.
        '''
    
        while True:    
            try:
                self.n = int(input("Please provide the number of participants: "))

                if (self.n%10 == 0) and (self.n < 2700) and (self.n >= 10):
                    break

                print("Number of participants needs to be dividable by 10 and smaller than 2700!")
                   
            except:
                print("Please provide an integer value!\n")
        
    def names(self):
        
        '''
            Args: None
            Returns: list of length self.n including a random mixture of male and female names
        '''

        #load names from lists
        with open("names\german-names-female.txt", encoding='utf8') as f:
            names_female = f.read().splitlines() 

        with open("names\german-names-male.txt", encoding='utf8') as f:
            names_male = f.read().splitlines() 

        # set percentages
        p_female = 50
        p_male = 50

        # create sublists of unique names
        l_female = random.sample(names_female,k = int(self.n/100 * p_female))
        l_male = random.sample(names_male,k = int(self.n/100 * p_male))

        # unify and shuffle
        l_name = l_female + l_male 
        random.shuffle(l_name)
        
        return l_name
        
    def create_basic(self):
        
        '''
            Args: None
            Returns: pd.dataframe including the following features:
                        ['ID', 'Name', 'Preferred language', 'Majors', 'Level of ambition']
            
                     string indicating the nature of the returned dataframe ("basic")
        '''

        # 1: Create IDs
        l_id = list(range(1,self.n+1))

        # 2: create names 
        
        l_name = self.names()

        # 3: create language preferences

        # set percentages
        p_any = 80
        p_en =  10
        p_ger = 10

        # create sublists
        l_any = ["Any"] * int(self.n/100 * p_any)
        l_en = ["English"] * int(self.n/100 * p_en)
        l_ger = ["German"] * int(self.n/100 * p_ger)

        # unify and shuffle
        l_lang = l_any + l_en + l_ger 
        random.shuffle(l_lang)

        # 4: create majors 

        maj = ["AI", "NP", "PHIL", "CL", "NI", "NB", "DS"]

        # create sublists

        maj_1 = random.choices(maj, k=self.n*2)
        maj_2 = random.choices(maj, k=self.n*2)

        # zip them into list of tuples
        tmp = list(zip(maj_1,maj_2))

        # remove dups
        l_maj = []

        for x in tmp:
            if x[0] != x[1]:
                l_maj.append(x)
            if len(l_maj) == self.n:
                break

        # 5: create ambitions 
        amb = ["Very low","Low","Medium","High","Very high"]
        l_amb = random.choices(amb, k=self.n)
        
        # save as dataframe
        df_basic = pd.DataFrame(list(zip(l_id, l_name, l_lang, l_maj, l_amb)), columns = [
                                                                                    'ID', 'Name', 'Preferred language', 
                                                                                    'Majors', 'Level of ambition'
                                                                                    ])

        return df_basic,  "basic"
            
    def create_full(self):
        
        '''
            Args: None
            Returns: pd.dataframe including the following features:
                        ['ID', 'Name', 'Preferred language', 'Majors', 'Level of ambition',
                        'Prefered meeting place', 'Personality type', 'Best friend',
                        'Openness', 'Blocked day]
            
                     string indicating the nature of the returned dataframe ("full")
        '''
        
        df_basic, _ = self.create_basic()
        
        # 1: Meeting place

        # set percentages
        p_online = 20
        p_ip =  80

        # create sublists
        l_online = ["Online"] * int(self.n/100 * p_online)
        l_ip = ["In person"] * int(self.n/100 * p_ip)

        # unify and shuffle
        l_meet = l_online + l_ip
        random.shuffle(l_meet)

        # 2: Personality type

        pers = [
            "ESTJ", "ENTJ", "ESFJ", "ENFJ", 
            "ISTJ", "ISFJ", "INTJ", "INFJ", 
            "ESTP", "ESFP", "ENTP", "ENFP", 
            "ISTP", "ISFP", "INTP", "INFP"
            ]

        l_pers = random.choices(pers, k=self.n)

        # 3: Best friend(s)

        l_bf = df_basic['Name'].tolist()

        l_bf_1 = l_bf[:int(len(l_bf)/2)]
        l_bf_2 = l_bf[int(len(l_bf)/2):]
        random.shuffle(l_bf_1)
        random.shuffle(l_bf_2)

        l_friends = []
        for i in range(len(l_bf)):
            try:
                ind = l_bf_1.index(l_bf[i])
                l_friends.append(l_bf_2[ind])
            except:
                ind = l_bf_2.index(l_bf[i])
                l_friends.append(l_bf_1[ind])

        # 4: Openness towards new people

        # set percentages
        p_rel = 20
        p_neu = 40
        p_con = 40

        #create sublists
        l_rel = ["Reluctant"] * int(self.n/100 * p_rel)
        l_neu = ["Neutral"] * int(self.n/100 * p_neu)
        l_con = ["Confident"] * int(self.n/100 * p_con)

        # unify and shuffle
        l_open = l_rel + l_neu + l_con
        random.shuffle(l_open)

        # 5: Timetable
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        l_days = []
        for day in days:
            l_days = l_days + ([day] * int(self.n/100 * 20))
        random.shuffle(l_days)
        
        # create dataframe

        df_extra = pd.DataFrame(list(zip(l_meet, l_pers, l_friends, l_open, l_days)), columns = [
                                                                                        'Prefered meeting place', 'Personality type', 
                                                                                        'Best friend', 'Openness', 'Blocked day'
                                                                                        ])

        return (pd.concat([df_basic, df_extra],axis=1)), "full"