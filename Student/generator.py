import pandas as pd
import random
import os  
import copy
import math
import itertools
import argparse
from sys import platform

class DataGenerator():
    
    '''
        This class serves to create a dataset in the form of a pd.dataframe. 
        This dataframe can either include basic features or all available 
        features depending on which internal method is called.  
    '''

    def __init__ (self,*size):
        
        '''
            If not already provided through arguments, this prompts user input
            with regards to the desired sample size. Said input needs to be of
            type integer, dividable by 10 and of size 10 - 2700.
        '''
        
        # check if size is provided
        
        try:
            self.n = size[0]
            
        except:
            
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
                     list of length self.n including matching genders and sometimes indeterminate at random (m/f/x)
        '''

        #load names from lists
        if platform == "win32":
            path = chr(92)
        else:
            path = "/" 

        with open("names" + path + "german-names-female.txt", encoding='utf8') as f:
            names_female = f.read().splitlines() 

        with open("names" + path + "german-names-male.txt", encoding='utf8') as f:
            names_male = f.read().splitlines() 

        # set percentages
        p_female = 50
        p_male = 50

        # create sublists of unique names
        l_female = random.sample(names_female,k = int(self.n/100 * p_female))
        l_male = random.sample(names_male,k = int(self.n/100 * p_male))
        
        l_female = list(zip(l_female, itertools.repeat("Female")))
        l_male = list(zip(l_male, itertools.repeat("Male")))

        # unify and shuffle
        l_tmp = l_female + l_male 
        random.shuffle(l_tmp)
        
        # unpack into gender and name lists
        l_name, l_gen = zip(*l_tmp)
        l_name = list(l_name)
        l_gen = list(l_gen)
        
        # create x genders (~5%)
             
        for i in range(int(self.n/100 * 5)):
            random_index = random.randrange(len(l_gen))
            l_gen[random_index] = "Indeterminate"
            
        return l_name, l_gen
        
    def create_basic(self):
        
        '''
            Args: None
            Returns: pd.dataframe including the following features:
                        ['ID', 'Name', 'Gender', 'Preferred language', 'Majors', 'Level of ambition']
            
                     string indicating the nature of the returned dataframe ("basic")
        '''

        # 1: Create IDs
        l_id = list(range(1,self.n+1))

        # 2: create names and genders
        
        l_name, l_gen = self.names()        

        # 4: create language preferences

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

        # 5: create majors 

        maj = ["AI", "CP", "CL", "NI", "NS", "PHIL"]

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
        df_basic = pd.DataFrame(list(zip(l_id, l_name, l_gen, l_lang, l_maj, l_amb)), columns = [
                                                                                    'ID', 'Name', 'Gender', 'Preferred language', 
                                                                                    'Majors', 'Level of ambition'
                                                                                    ])

        return df_basic,  "basic"
            
    def create_full(self):
        
        '''
            Args: None
            Returns: pd.dataframe including the following features:
                        ['ID', 'Name', 'Gender', Preferred language', 'Majors', 'Level 
                        of ambition','Preferred meeting place', 'Personality type', 
                        'Best friend','Openness', 'Blocked day]
            
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

        # 3: Best friend

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

        # 4: Timetable
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        l_days = []
        for day in days:
            l_days = l_days + ([day] * int(self.n/100 * 20))
        random.shuffle(l_days)
        
        # create dataframe

        df_extra = pd.DataFrame(list(zip(l_meet, l_pers, l_friends, l_days)), columns = [
                                                                                        'Preferred meeting place', 'Personality type', 
                                                                                        'Best friend', 'Preferred day'
                                                                                        ])

        return (pd.concat([df_basic, df_extra],axis=1)), "full"
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Dataset generation for the Genetic Algorithms task")
    
    parser.add_argument("size", default="50", 
                        type=int, help='Should be dividable by 10 and lower than 2700')
    parser.add_argument("type", default="full", 
                        type=str, choices=["basic", "full"],
                        help='Choose from either the basic or the full featured dataset')
  
    args = parser.parse_args()
    
    generator = DataGenerator(args.size)
    
    if args.type == "basic":
        dataset, type_ = generator.create_basic()
    else: 
        dataset, type_ = generator.create_full()
        
    # save dataframe to csv

    dataset.to_csv("dataset_" + type_ + ".csv", encoding="utf-8-sig", index=False)
