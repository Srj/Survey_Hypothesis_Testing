import numpy as np
import pandas as pd
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

DATA = 'data/'
def preprocess():
    print("Loading Data...")
    df = pd.read_csv(os.path.join(DATA,'Survey.csv'))
    print('Loading Done...')
    #Fixing Sex Column
    df.sex.replace({'Transgender 1': 'Transgender','0nbinary':'Binary','1':'Female','2':'Male'},inplace=True)

    #Drop 2 Outliers
    df = df[df['sex'].isin(['Female','Male'])]

    #Fixing Profession
    df.Profession.replace({'others':'Others'},inplace=True)

    #Fixing Country
    df.Country.replace({'Maxico':'Mexico'},inplace=True)

    #Fixing Education
    df['@4.HighestLevelofeducation'].replace({'Bachelor/undergraduate':'T','Bachelor\'s degree': 'T',
                                            'Master/graduate or Ph.D.':'T','Junior high school or lower':'P',
                                            'High school':'S','PhD':'T','Masters':'T','college':'S',
                                            'master\'s degree':'T','Bachelor':'T','Master':'T',
                                            'Graduate School(Master) å¤§å­¦é™¢å’æ¥­ï¼ˆä¿®å£«ï¼‰':'T',
                                            'Doctorate':'T','bachelor':'T','Bachelor å¤§å­¦å’æ¥­':'T',
                                            'high school':'S','HSC':'S','University':'T',
                                            'Some college':'S','Master\'s degree':'T','Tertiary':'T',
                                            'Bachelors':'T','High School é«˜ç­‰å­¦æ ¡å’æ¥­':'S',
                                            'Graduation':'T','Graduate': 'T','Undergraduate':'T',
                                            'High School é«˜ç‰å¦æ ¡å’æ¥':'S','High school diploma':'S',
                                            'Bachelor\'s':'T','Masters Degree':'T','Bachelor degree':'T',
                                            'Bsc':'T','Masters degree':'T','Ho0urs':'T',
                                            'Bachelors Degree':'T','MSc':'T','Mbbs':'T','Graduate School(Doctor) å¤§å­¦é™¢å’æ¥­ï¼ˆåšå£«ï¼‰':'T',
                                            'MBBS 2nd year':'T','M.B.B.S':'T','Hsc':'S','MSN':'T','MBBS 1t year':'T'
                                            ,'MBBS':'T','High School':'T','College':'T','Degree':'T','High School é«˜ç\xad‰å\xad¦æ\xa0¡å\x8d’æ¥\xad':'S'
                                            ,'Diploma':'T','Phd':'T','High School Diploma':'T','Specialist':'T','MPH':'T','MD':'T',
                                            'Bachelorâ€™s degree':'T','Mphil':'T','MS':'T','Postgraduate':'T','Bachelor Degree':
                                            'T','Post graduation':'T','BSN':'T'},inplace=True)
    df['Education'] = df[df['@4.HighestLevelofeducation'].isin(['P','S','T'])]['@4.HighestLevelofeducation']

    #Drop Age = 2,1885,1984
    df = df[~df['Age'].isin([2,1885,1984])]
    #Binning by Age
    df['AgeBin'] = pd.qcut(df['Age'],5)
    print('Fixing Error Done...')

    #Creating Processed Dataframe
    cleaned = pd.DataFrame()
    cleaned['Profession'] = df['Profession']
    cleaned['Country'] = df['Country']
    cleaned['Region'] = df['Region']
    cleaned['Education'] = df['Education']
    cleaned['Sex'] = df['sex']
    cleaned['Age'] = df['Age']
    cleaned['AgeBin'] = df['AgeBin']
    cleaned['Maritial'] = df['@8.Maritalstatus']

    cleaned['1.1_HeardCovid'] = df['A1.1HaveyouheardaboutCOVID19']

    cleaned['1.2_Television/Radio'] = df['A1.2 Where did you hear about COVID-19 most? (check all that apply ) [Television/Radio]']
    cleaned['1.2_Newspaper/Magazines'] = df['A1.2 Where did you hear about COVID-19 most? (check all that apply ) [Newspaper/ Magazines]']
    cleaned['1.2_SocialMedia'] = df['WheredidyouhearaboutCOVID19mostcheckallthatapplySocialmedia']
    cleaned['1.2_Colleagues/Workplace'] = df['A1.2 Where did you hear about COVID-19 most? (check all that apply ) [Colleagues/workplace]']
    cleaned['1.2_Neighbors'] = df['A1.2 Where did you hear about COVID-19 most? (check all that apply ) [Neighbors]']

    cleaned['1.3_CovidKnowledgeLevel'] = df['B1.3 How would you rate the extend of your k0wledge of COVID-19?']

    cleaned['1.4_ContactRespiratoryDrop'] = df['B1.4 How does COVID19 Spread/Transmitted (check all that apply) [Contact with respiratory droplets]']
    cleaned['1.4_Touching'] = df['B1.4 How does COVID19 Spread/Transmitted (check all that apply) [Touching and shaking hands with an infected person]']
    cleaned['1.4_UseSameObject'] = df['B1.4 How does COVID19 Spread/Transmitted (check all that apply) [The use of objects used by an infected person]']
    cleaned['1.4_Sex'] = df['B1.4 How does COVID19 Spread/Transmitted (check all that apply) [Sexual route]']
    cleaned['1.4_PersonToPerson'] = df['B1.4 How does COVID19 Spread/Transmitted (check all that apply) [Person-to-person]']
    cleaned['1.4_CloseContact'] = df['B1.4 How does COVID19 Spread/Transmitted (check all that apply) [Close contact]']
    cleaned['1.4_TouchingCoin'] = df['B1.4 How does COVID19 Spread/Transmitted (check all that apply) [Touching currency/Coin]']
    cleaned['1.4_CovidFloatOnAir'] = df['B1.4 How does COVID19 Spread/Transmitted (check all that apply) [COVID-19 can float on air almost 30 minutes]']

    cleaned['1.5_Fever'] = df['B1.5 In your opinion, what are the signs and symptoms of COVID-19 (check all that apply) [Fever]']
    cleaned['1.5_Tiredness'] = df['B1.5 In your opinion, what are the signs and symptoms of COVID-19 (check all that apply) [Tiredness]']
    cleaned['1.5_Cough'] = df['B1.5 In your opinion, what are the signs and symptoms of COVID-19 (check all that apply) [Dry cough]']
    cleaned['1.5_ShortnessBreath'] = df['B1.5 In your opinion, what are the signs and symptoms of COVID-19 (check all that apply) [Shortness of breath/Breathing difficulties]']
    cleaned['1.5_AchesPain'] = df['B1.5 In your opinion, what are the signs and symptoms of COVID-19 (check all that apply) [aches and pains]']
    cleaned['1.5_NasalCongestion'] = df['B1.5 In your opinion, what are the signs and symptoms of COVID-19 (check all that apply) [nasal congestion]']
    cleaned['1.5_RunningNose'] = df['B1.5 In your opinion, what are the signs and symptoms of COVID-19 (check all that apply) [runny 0se]']
    cleaned['1.5_SoreThroat'] = df['B1.5 In your opinion, what are the signs and symptoms of COVID-19 (check all that apply) [sore throat]']
    cleaned['1.5_Diarrhea'] = df['B1.5 In your opinion, what are the signs and symptoms of COVID-19 (check all that apply) [Diarrhea]']
                                
    cleaned['1.6_Mask'] = df['B1.6 Which mask(s) do you think is best to control the spread of the COVID-19?']

    cleaned['1.7_IncubationPeriod'] = df['B1.7 How long is the incubation period for COVID-19?']

    cleaned['1.8_Vaccine'] = df['B1.8 Are there any vaccines, drugs or treatments for COVID-19?']

    cleaned['1.9_LockDown'] = df['C1.9 Please check for which you familiar with? (check all that apply) [lock-down]']
    cleaned['1.9_Isolation'] = df['C1.9 Please check for which you familiar with? (check all that apply) [self-isolation]']
    cleaned['1.9_Quarantine'] = df['C1.9 Please check for which you familiar with? (check all that apply) [home quarantine]']

    cleaned['1.10_Over60Yr'] = df['C1.10 Is anyone in your immediate environment at risk of infection with COVID-19 (e.g. parents, siblings, close friends/colleagues) due to the following? (please select all relevant responses) [age (over 60 years)]']
    cleaned['1.10_Over60Yr'].replace({'0 estoy seguro':'1'},inplace=True)
    cleaned['1.10_PrexistingCond'] = df['C1.10 Is anyone in your immediate environment at risk of infection with COVID-19 (e.g. parents, siblings, close friends/colleagues) due to the following? (please select all relevant responses) [severe pre-existing conditions (e.g. Respiratory diseases, heart diseases, cancer, immune deficiency problem)]']
    cleaned['1.10_PrexistingCond'].replace({'If 0':'0'},inplace=True)
    cleaned['1.10_WorkEnvo'] = df['C1.10 Is anyone in your immediate environment at risk of infection with COVID-19 (e.g. parents, siblings, close friends/colleagues) due to the following? (please select all relevant responses) [work environment (e.g. Working in the health care environment and involving in contact with patients)]']
    cleaned['1.10_WorkEnvo'].replace({'If 0':'0','0, 3':'3'},inplace=True)
    cleaned['1.10_WorkEnvo'].value_counts()
    cleaned['1.10_RiskZone'] = df['C1.10 Is anyone in your immediate environment at risk of infection with COVID-19 (e.g. parents, siblings, close friends/colleagues) due to the following? (please select all relevant responses) [exposure to a risk zone/areas/countries (e.g. China, Italy etc.)]']
    cleaned['1.10_RiskZone'].replace({'If 0':'0','0,3':'3','0, 3':'3','0 3':'3','0,  3':'3','I\'m 0 sure':'1','1. 0':'0'},inplace=True)
    cleaned['1.10_RiskZone'].value_counts()

    cleaned['2.1_LockDown'] = df['B2.1 Do you think the government should lock-down/restrict travel areas to avoid spread of COVID-19?']
    cleaned['2.1_LockDown'].replace({'I 2':'2'},inplace=True)
    cleaned['2.2_HomeQuarantine'] = df['B2.2 Do you think home quarantine can reduce COVID-19 outbreaks?']
    cleaned['2.2_HomeQuarantine'].replace({'I 2':'2'},inplace=True)
    cleaned['2.3_Isolation'] = df['B2.3 Isolation and treatment of infected people are effective ways to reduce the spread of the virus?']
    cleaned['2.4_PersonalHygiene'] = df['B2.4 Do you think personal hygiene is important in controlling the spread of COVID-19?']
    cleaned['2.5_MediaRole'] = df['B2.5 Media should take a leading role in raising awareness coronavirus risk reduction and prevention issues ?']
    cleaned['2.5_MediaRole'].replace({'Totalmente en desacuerdo':'1','I 2': '2'},inplace=True)
    cleaned['2.5_MediaRole'].value_counts()

    cleaned['2.6_Over60Yr'] = df['C2.6 Do you think you are at increased personal risk of infection with COVID-19 due to any of the following?(please select all relevant responses) [age (over 60 years)]']
    cleaned['2.6_Over60Yr'].replace({'I don\'t k0w / 0 sure':'Not Sure','0;I don\'t k0w / 0 sure':'Not Sure',
                                '1;I don\'t k0w / 0 sure':'Not Sure'},inplace = True)
    cleaned['2.6_Over60Yr'].value_counts()

    cleaned['2.6_PreExisting'] = df['C2.6 Do you think you are at increased personal risk of infection with COVID-19 due to any of the following?(please select all relevant responses) [severe pre-existing conditions (e.g. Respiratory diseases, heart diseases, cancer, immune deficiency problem)]']
    cleaned['2.6_PreExisting'].replace({'I don\'t k0w / 0 sure':'Not Sure','0;I don\'t k0w / 0 sure':'Not Sure',
                                '1;I don\'t k0w / 0 sure':'Not Sure'},inplace = True)
    cleaned['2.6_PreExisting'].value_counts()

    cleaned['2.6_Working'] = df['C2.6 Do you think you are at increased personal risk of infection with COVID-19 due to any of the following?(please select all relevant responses) [work environment (e.g. Working in the health care environment and involving in contact with patients)]']
    cleaned['2.6_Working'].replace({'I don\'t k0w / 0 sure':'Not Sure','0;I don\'t k0w / 0 sure':'Not Sure',
                                '1;I don\'t k0w / 0 sure':'Not Sure','0, I don\'t k0w / 0 sure':'Not Sure'},inplace = True)
    cleaned['2.6_Working'].value_counts()

    cleaned['2.6_RiskZone'] = df['C2.6 Do you think you are at increased personal risk of infection with COVID-19 due to any of the following?(please select all relevant responses) [exposure to a risk zone/areas/countries (e.g. China, Italy etc.)]']
    cleaned['2.6_RiskZone'].replace({'I don\'t k0w / 0 sure':'Not Sure','0;I don\'t k0w / 0 sure':'Not Sure',
                                '1;I don\'t k0w / 0 sure':'Not Sure','0, I don\'t k0w / 0 sure':'Not Sure'},inplace = True)
    cleaned['2.6_RiskZone'].value_counts()

    cleaned['3.1_Quarantine'] = df['D3.1 Do you have any of the following practices to prevent COVID-19 transmission (check all that apply)? [Practicing self-isolation/Home quarantine]']
    cleaned['3.1_Quarantine'].value_counts()
    cleaned['3.1_Respository'] = df['D3.1 Do you have any of the following practices to prevent COVID-19 transmission (check all that apply)? [Practicing respiratory hygiene]']
    cleaned['3.1_Respository'].value_counts()
    cleaned['3.1_WashingHand'] = df['D3.1 Do you have any of the following practices to prevent COVID-19 transmission (check all that apply)? [washing hand frequently using hand sanitizer (alcohol based)]']
    cleaned['3.1_WashingHand'].value_counts()
    cleaned['3.1_Mask'] = df['D3.1 Do you have any of the following practices to prevent COVID-19 transmission (check all that apply)? [Using face mask (Surgical)]']
    cleaned['3.1_Mask'].value_counts()
    cleaned['3.1_TouchingMouth'] = df['D3.1 Do you have any of the following practices to prevent COVID-19 transmission (check all that apply)? [Avoiding touching 0se, mouth and e1]']
    cleaned['3.1_TouchingMouth'].value_counts()
    cleaned['3.1_SocialDistance'] = df['D3.1 Do you have any of the following practices to prevent COVID-19 transmission (check all that apply)? [Maintaining social distance (min 1 meter)]']
    cleaned['3.1_SocialDistance'].value_counts()

    cleaned['3.2_PPE'] = df['D3.2 Have you been provided with personal protection equipment (PPE) in your workplace?']
    cleaned['3.2_PPE'].replace({'3 ã‚ã‹ã‚‰ãªã„':'3'},inplace=True)
    cleaned['3.2_PPE'].value_counts()

    cleaned['3.3_Test'] = df['D3.3 Have you tested yourself for COVID-19?']
    cleaned['3.3_Test'].replace({'1, it was requested through my medical health service':'1',},inplace=True)
    cleaned.loc[~cleaned['3.3_Test'].isin(['0','1','3']),'3.3_Test'] = '3'
    cleaned['3.3_Test'].value_counts()

    cleaned['3.4_Handshake'] = df['D3.4 Do you have any of the following practices 1w a days? [any handshake?] Neg']
    cleaned['3.4_Handshake'].value_counts()
    cleaned['3.4_Hug'] = df['D3.4 Do you have any of the following practices 1w a days? [Hug?]Neg']
    cleaned['3.4_Hug'].value_counts()
    cleaned['3.4_VisitPlaces'] = df['D3.4 Do you have any of the following practices 1w a days? [Visiting public places?]Neg']
    cleaned['3.4_VisitPlaces'].value_counts()
    cleaned['3.4_ContactInfected'] = df['D3.4 Do you have any of the following practices 1w a days? [contact with infected person?]Neg']
    cleaned['3.4_ContactInfected'].value_counts()
    cleaned['3.4_Religion'] = df['D3.4 Do you have any of the following practices 1w a days? [You/your family members going church/ mosque/ temple/synagogue/ pagoda for prayer]Neg']
    cleaned['3.4_Religion'].replace({'IF 1T':'1'},inplace=True)
    cleaned['3.4_Religion'].value_counts()

    cleaned['3.5_HandWash'] = df['D3.5 how many times did you wash hand in last 12 hours or 12 hours?']

    cleaned['3.8_Internet'] = df['A3.8 Do you or your household members use internet?']

    cleaned['3.9_Newspaper'] = df['A3.9 Please indicate which of the following do you use for COVID-19 update (check all that apply)? [Newspaper]']
    cleaned['3.9_TV'] = df['A3.9 Please indicate which of the following do you use for COVID-19 update (check all that apply)? [TV (local/ international)]']
    cleaned['3.9_SocialMedia'] = df['A3.9 Please indicate which of the following do you use for COVID-19 update (check all that apply)? [Social media (Facebook, Instagram, Line, YouTube etc.)]']
    cleaned['3.9_Internet'] = df['A3.9 Please indicate which of the following do you use for COVID-19 update (check all that apply)? [Internet (WHO websites)]']

    cleaned['3.10_TimeSpentOnCovidNews'] = df['A3.10 How much time, on average, per day do you spend on the topics related to COVID-19 (e.g. via news coverage, work, conversations, thoughts)? Please indicate a daily average..']

    cleaned['4.1_FearOfDeath'] = df['E4.1 What are your mental health/psychological problems regarding COVID-19? (check all that apply) [Fear of falling ill and dying]']
    cleaned['4.1_FearOfDeath'].value_counts()
    cleaned['4.1_Anxeity'] = df['E4.1 What are your mental health/psychological problems regarding COVID-19? (check all that apply) [Anxiety]']
    cleaned['4.1_Anxeity'].value_counts()
    cleaned['4.1_Depression'] = df['E4.1 What are your mental health/psychological problems regarding COVID-19? (check all that apply) [Depression]']
    cleaned['4.1_Depression'].value_counts()
    cleaned['4.1_SocialExclusion'] = df['E4.1 What are your mental health/psychological problems regarding COVID-19? (check all that apply) [Fear of being socially excluded/placed in quarantine]']
    cleaned['4.1_SocialExclusion'].value_counts()
    cleaned['4.1_Lonliness'] = df['E4.1 What are your mental health/psychological problems regarding COVID-19? (check all that apply) [Feelings of helplessness, boredom, loneliness]']
    cleaned['4.1_Lonliness'].value_counts()
    cleaned['1.5_Diarrhea'].replace({'If 0':'0'},inplace=True)
    cleaned['1.5_SoreThroat'].replace({'If 0':'0'},inplace=True)
    cleaned['1.5_RunningNose'].replace({'If 0':'0'},inplace=True)
    cleaned['1.5_NasalCongestion'].replace({'If 0':'0'},inplace=True)
    cleaned['1.5_AchesPain'].replace({'If 0':'0'},inplace=True)
    cleaned['1.5_Cough'].replace({'If 0':'0'},inplace=True)
    cleaned['1.5_Tiredness'].replace({'If 0':'0'},inplace=True)
    cleaned['1.3_CovidKnowledgeLevel'].replace({'Very limited':'1','Good understanding':'5','一般':'3',
                                                'Familiar':'3','Unfamiliar':'1','了解':'5','非常不了解':'1',
                                            '不了解':'1','非常了解':'5'},inplace=True)
    cleaned['3.9_Internet'].replace({'IF 0T':'0'},inplace=True)
    cleaned['3.9_TV'].replace({'IF 0T':'0'},inplace=True)
    cleaned['3.9_Newspaper'].replace({'IF 0T':'0'},inplace=True)
    cleaned['3.8_Internet'].replace({'0t':'0'},inplace=True)
    cleaned['N90BestToControlSpread'] = df['N90 mask(s) do you think is best to control the spread of the COVID-19? (If yes then 1 else 0)']
    print('Preprocess Done...')

    cleaned['1.7_IncubationPeriod'].replace({'2-14 days' : '2-14 Day','2-14days':'2-14 Day','Don\'t k0w/3':'Dont Know',
                                            '1-7 days':'1-7 Day','1-7days':'1-7 Day','Dont k0w/ 3':'Dont Know',
                                            '7 a 14 dias': 'Other','Up to 28days or even longer':'Other',
                                            'From 1 to 14days ..some time 15 or 17 days':'Other','11 days':'Other',
                                            'It could be since the first day of syntoms along day 28.':'Other',
                                            'I heard and read as 14 days. 3 though.':'Other',
                                            'Up to 5 days to show symptoms, but some are outliers, so 2-14 days is the guideline.\
                                            Some recorded after 20 plus days, but may be due to faulty testing or re-infection':'Other',    
    },inplace=True)
    cleaned.loc[~cleaned['1.7_IncubationPeriod'].isin(['2-14 Day','Dont Know','1-7 Day']),'1.7_IncubationPeriod'] = 'Other'
    cleaned['1.7_IncubationPeriod'].value_counts()
    cleaned['1.9_Isolation'].replace({'If 0' : '0'},inplace=True)

    #Defing Metrics
    cleaned['LearntCovid'] = cleaned['1.2_Television/Radio'] \
                                    + cleaned['1.2_Newspaper/Magazines'] + cleaned['1.2_SocialMedia'] +\
                                    cleaned['1.2_Colleagues/Workplace'] + cleaned['1.2_Neighbors'] 
    cleaned['KnowledgeScore'] = df['Knowledge scoring {22/22}']  # Some value greater than 22
    cleaned['OpinionScoring'] = df['Opinion-scoring {25/25}']
    cleaned['Susceptiblity'] = df['Susceptibility-scoring {4/4}']
    cleaned['BehaveScore'] = df['Behaviour-scoring {11/11}']
    cleaned['PsychologicalHealth'] = df['Psychological-health Soring{5/5}']
    cleaned['Access_Of_Information'] = cleaned['3.9_Newspaper'].fillna(0).astype(int) + cleaned['3.9_TV'].fillna(0).astype(int) +cleaned['3.9_SocialMedia'].fillna(0).astype(int) + cleaned['3.9_Internet'].fillna(0).astype(int) 
    print('Defined Metrics')
    cleaned.to_csv(os.path.join(DATA,'Cleaned.csv'),index=False)

preprocess()                       