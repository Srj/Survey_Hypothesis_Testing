import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import dataframe_image as dfi
import warnings
import math
import os
plt.rcParams['figure.figsize']= (12,12)  
plt.style.use('fivethirtyeight')
#Size for x and y ticks
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)
warnings.filterwarnings('ignore')
def country_code(x):   # A function to convert Full Country Name to ISO-2 Country Code
    if x == 'United States':
        return 'US'
    elif x == 'Japan':
        return 'JP'
    elif x == 'Bangladesh':
        return 'BD'
    elif x =='Zambia':
        return 'ZM'
    elif x =='Mexico':
        return 'MX'
    elif x =='Malaysia':
        return 'MY'
    elif x =='Pakistan':
        return 'PK'
    elif x == 'China':
        return 'CN'

# A function to calculate Chi Squared Test
def chi_squared_test(df,country,column1,column2,prob=0.95):
    """
    df: Dataframe containing necessary columns
    country: which country to analyze
    column1: which field to snalyze
    column2: which metric/criterion to analyze
    prob: significance of test i.e, 0.95,.99 etc
    
    returns
    country: The country analyzed
    stat: Chi Squared Test statistics
    critical: critical Value for this test
    p: p value for this test
    p < (1 - prob) : returns true if there is significant differences
    """
    from scipy import stats
    df = df[df['Country'] == country]
    contingency_table = pd.crosstab(df[column1],df[column2])
    try:
        stat,p,dof,expected = stats.chi2_contingency(contingency_table)
    except:
        return country, None,None,None,None
    critical = stats.chi2.ppf(prob,dof)
    contingency_table.plot.bar(stacked=False)
    plt.title(country,fontsize=100)
    plt.tight_layout()
    plt.savefig(f'Graph/{column2}/{column1}/{column2}_{column1}_{country}.png',dpi=80)
    return country, stat,critical , p , p < (1- prob)        


def analyse():
    cleaned = pd.read_csv('data/Cleaned.csv')  # Importing CSV File
    countries = cleaned.Country.value_counts().index[:8].tolist() 
    cleaned =cleaned[cleaned['Country'].isin(countries)]
    print('Removed Low Frequency Countries')

    # Apply country_code function to Convert Country column to ISO-2 Code and save into a new Column named C_Code
    cleaned['C_Code'] = cleaned['Country'].apply(lambda x :country_code(x)) 

    #Binning Different Ages in Age Group. Here We have 3 groups [0,25), [25,50), [50,inf). 
    #This method is left inclusive
    bins = pd.IntervalIndex.from_tuples([(0,25),(25,50),(50,np.inf)])

    #Now Applying binning and saving in AgeBin Column
    cleaned['AgeBin'] = pd.cut(cleaned.Age,bins)

    #Dropping Some other complicated Low resources samples. Don't Bother with it
    cleaned = cleaned[~((cleaned['Country'].isin(['Bangladesh','Pakistan'])) & (cleaned['Age'] > 50))]
    cleaned = cleaned[~((cleaned['Country'].isin(['Mexico'])) & (cleaned['Education'] =='P'))]

    #The Metrics We are analyzing
    criterion = ['Access_Of_Information', 'KnowledgeScore', 'OpinionScoring',
        'Susceptiblity', 'BehaveScore', 'PsychologicalHealth']

    #Feature of Sample
    sub_field = ['AgeBin','Education','Profession','Sex']

    #This complicated Loop with calculcate Mean, Count, 90% CI, 95 CI% and 
    #save each in {criterion}_{subfield}_table.csv file. 
    for col in criterion:
        for f in sub_field:
            print(col,f)
            stats = cleaned.groupby(['Country',f])[col].agg(['mean', 'count', 'std'])

            ci95_hi = []
            ci95_lo = []
            ci90_lo = []
            ci90_hi = []

            for i in stats.index:
                m, c, s = stats.loc[i]
                ci95_hi.append(m + 1.95*s) # 95% CI
                ci95_lo.append(m - 1.95*s)
                
                ci90_hi.append(m + 1.65*s) # 90% CI
                ci90_lo.append(m - 1.65*s)

            stats['CI95_high'] = ci95_hi
            stats['CI95_low'] = ci95_lo
            stats['CI90_high'] = ci90_hi
            stats['CI90_low'] = ci90_lo
            #These prohibits low interval from being negative i.e (-1.5,4) will be converted to (0,4).
            #Delete following two lines to allow negative values.
            stats['CI95_low'] = stats['CI95_low'].apply(lambda x: max(0,x))
            stats['CI90_low'] = stats['CI90_low'].apply(lambda x: max(0,x))
            
            stats.rename(columns={'mean':'Mean','count':'Count','std':'Std.'},inplace=True)
            #Saving in CSV
            stats.to_csv(f'data/{col}_{f}_table.csv',index=True,float_format='%.3f')
    print('Generated Stats...')
    #Create Necessary Dirs
    #Frequency Graphs will be saved in {Criterion}/{Field} subdir
    criterion = ['Access_Of_Information','BehaveScore', 'KnowledgeScore', 'OpinionScoring' , 'PsychologicalHealth','Susceptiblity',]
    field = ['AgeBin','Education','Profession','Sex',]
    for c in criterion:
        for f in field:
            if not os.path.exists(os.path.join('Graph',c,f)):
                os.makedirs(os.path.join('Graph',c,f))

    print('Created Graph Directories')

    p_df = pd.DataFrame()
    # A loop to calculate all criterion over all fields.
    for c in criterion:
        for f in field:
            results = []
            for i, country in enumerate(countries):
                results.append(chi_squared_test(cleaned,country,f,c))
            output = pd.DataFrame(results, columns = ['Country','Value', 'Critical','p-value','critical'])
            output['Significant'] = output['p-value'] < 0.05
            output = output.sort_values(['Country'])
            if len(p_df) == 0:
                p_df = pd.DataFrame(output[['Country','p-value']])
            else:
                p_df = pd.merge(p_df,output[['Country','p-value']],on='Country',suffixes=(None,c + '_' + f))
            output[['Country','p-value','Significant']].to_csv(f'data/{c}_{f}_test.csv',index= False,float_format = '%.3f')
            df_styled = output.style.background_gradient()
            # print(c+ " " + f)
            # print(output)
            dfi.export(df_styled,f"Graph/{c}/{f}/table.png")

    print('Chi Square Result Generated...')

    #Save p-values in CSV file
    p = p_df.T
    p.columns = p.loc['Country']
    p.drop(['Country'],inplace=True)
    p = p.astype(float)
    p.to_csv('data/p_value.csv',float_format='%.6f')


    fig,axs = plt.subplots(2,4)
    # axs = axs.flatten()
    criterion = ['Access_Of_Information', 'KnowledgeScore', 'OpinionScoring',
        'Susceptiblity', 'BehaveScore', 'PsychologicalHealth', 'LearntCovid']
    countries_list = [['Mexico','United States','Bangladesh','Pakistan','China'],
                    ['Mexico','United States','Bangladesh','China'],
                    ['Mexico','United States','Bangladesh','Malaysia','Pakistan','China'],
                    ['Mexico','United States','Bangladesh','Pakistan','Malaysia','Zambia','Japan','China']]
    label_list = [['MX','US','BD','PK','CN'],
                ['MX','US','BD','CN'],
                ['MX','US','BD','MY','PK','CN'],
                ['MX','US','BD','PK','MY','ZM','JP','CN']]
    sub_criterion = ['PsychologicalHealth','Susceptiblity']
    sub_titles = ['CovPsy','CovSus']
    ylim = [4,4]
    sub_field = ['AgeBin','Education','Profession','Sex']
    sub_field_title = ['AgeBin','Education','Profession','Gender']

    # cleaned =df[df['Country'].isin(countries)]
    for i,c in enumerate(sub_criterion):
        for j,f in enumerate(sub_field):
    #         if i == 0:
    #             temp = df[df['Country'].isin([x for x in countries_list[j] if x !='China'])]
    #         else:
            temp = cleaned[cleaned['Country'].isin(countries_list[j])]
            g= sns.barplot(x="C_Code", y=c,hue=f, data=temp,ci=95,errwidth=0.5,ax=axs[i,j])
            g.set(ylim=(0, ylim[i]))
    #         if i == 0:
    #             g.set_xticklabels([x for x in label_list[j] if x != 'CN'])
    #         else:
    #         g.set_xticklabels(label_list[j])
            g.set_xlabel(sub_field_title[j])
            if j != 0: 
                g.set_yticklabels([])          
                g.set_ylabel(None)
            else:
                g.set_ylabel(sub_titles[i])
            
            if i == 0:
                plt.legend()
            
            
    #         plt.tight_layout()
    #         plt.savefig(os.path.join('Graph',f'{c}_{f}.png'))
    for ax in fig.get_axes():
    #     ax.label_outer()
        ax.tick_params(axis='x',labelrotation=90)
    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.2)
    for ax in axs.flatten():    
        ax.legend(loc=4,fontsize=8)
        ax.grid(False)
    plt.figtext(0.5,0.5,"(a)", va="center", ha="center", size=15)
    plt.figtext(0.5,0.01,"(b)", va="center", ha="center", size=15)
    fig.tight_layout(pad=3.0)
    # for i in range(4):
    #     axs[0,i].legend(loc=0,fontsize=8)
    # axs[0,2].legend(loc=4,fontsize=8)
    # axs[1,3].legend(loc=4)
    plt.savefig('F3.png',dpi=350)

    
    
    fig,axs = plt.subplots(2,4)
    # axs = axs.flatten()
    criterion = ['Access_Of_Information', 'KnowledgeScore', 'OpinionScoring',
        'Susceptiblity', 'BehaveScore', 'PsychologicalHealth', 'LearntCovid']
    countries_list = [['Mexico','United States','Bangladesh','Pakistan','China'],
                    ['Mexico','United States','Bangladesh','China'],
                    ['Mexico','United States','Bangladesh','Malaysia','Pakistan','China'],
                    ['Mexico','United States','Bangladesh','Pakistan','Malaysia','Zambia','Japan','China']]
    label_list = [['MX','US','BD','PK','CN'],
                ['MX','US','BD','CN'],
                ['MX','US','BD','MY','PK','CN'],
                ['MX','US','BD','PK','MY','ZM','JP','CN']]
    sub_criterion = ['KnowledgeScore', 'OpinionScoring']
    sub_titles = ['CovKd','CovOp']
    ylim = [25,25]
    sub_field = ['AgeBin','Education','Profession','Sex']
    sub_field_title = ['AgeBin','Education','Profession','Gender']

    # cleaned =df[df['Country'].isin(countries)]
    for i,c in enumerate(sub_criterion):
        for j,f in enumerate(sub_field):
    #         if i == 0:
    #             temp = df[df['Country'].isin([x for x in countries_list[j] if x !='China'])]
    #         else:
            temp = cleaned[cleaned['Country'].isin(countries_list[j])]
            g= sns.barplot(x="C_Code", y=c,hue=f, data=temp,ci=95,errwidth=0.5,ax=axs[i,j])
            g.set(ylim=(0, ylim[i]))
    #         if i == 0:
    #             g.set_xticklabels([x for x in label_list[j] if x != 'CN'])
    #         else:
    #         g.set_xticklabels(label_list[j])
            g.set_xlabel(sub_field_title[j])
            if j != 0: 
                g.set_yticklabels([])          
                g.set_ylabel(None)
            else:
                g.set_ylabel(sub_titles[i])
            
            if i == 0:
                plt.legend()
            
            
    #         plt.tight_layout()
    #         plt.savefig(os.path.join('Graph',f'{c}_{f}.png'))
    for ax in fig.get_axes():
    #     ax.label_outer()
        ax.tick_params(axis='x',labelrotation=90)
    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.2)
    for ax in axs.flatten():    
        ax.legend(loc=4,fontsize=8)
        ax.grid(False)
    plt.figtext(0.5,0.5,"(a)", va="center", ha="center", size=15)
    plt.figtext(0.5,0.01,"(b)", va="center", ha="center", size=15)
    fig.tight_layout(pad=3.0)
    # for i in range(4):
    #     axs[0,i].legend(loc=0,fontsize=8)
    # axs[0,2].legend(loc=4,fontsize=8)
    # axs[1,3].legend(loc=4)
    plt.savefig('F2.png',dpi=350)


    fig,axs = plt.subplots(2,4)
    # axs = axs.flatten()
    criterion = ['Access_Of_Information', 'KnowledgeScore', 'OpinionScoring',
        'Susceptiblity', 'BehaveScore', 'PsychologicalHealth', 'LearntCovid']
    countries_list = [['Mexico','United States','Bangladesh','Pakistan','China'],
                    ['Mexico','United States','Bangladesh','China'],
                    ['Mexico','United States','Bangladesh','Malaysia','Pakistan','China'],
                    ['Mexico','United States','Bangladesh','Pakistan','Malaysia','Zambia','Japan','China']]
    label_list = [['MX','US','BD','PK','CN'],
                ['MX','US','BD','CN'],
                ['MX','US','BD','MY','PK','CN'],
                ['MX','US','BD','PK','MY','ZM','JP','CN']]
    sub_criterion = ['Access_Of_Information', 'BehaveScore']
    sub_titles = ['CovIA','CovBh']
    ylim = [4,12]
    sub_field = ['AgeBin','Education','Profession','Sex']
    sub_field_title = ['AgeBin','Education','Profession','Gender']

    # cleaned =df[df['Country'].isin(countries)]
    for i,c in enumerate(sub_criterion):
        for j,f in enumerate(sub_field):
    #         if i == 0:
    #             temp = df[df['Country'].isin([x for x in countries_list[j] if x !='China'])]
    #         else:
            temp = cleaned[cleaned['Country'].isin(countries_list[j])]
            g= sns.barplot(x="C_Code", y=c,hue=f, data=temp,ci=95,errwidth=0.5,ax=axs[i,j])
            g.set(ylim=(0, ylim[i]))
    #         if i == 0:
    #             g.set_xticklabels([x for x in label_list[j] if x != 'CN'])
    #         else:
    #         g.set_xticklabels(label_list[j])
            g.set_xlabel(sub_field_title[j])
            if j != 0: 
                g.set_yticklabels([])          
                g.set_ylabel(None)
            else:
                g.set_ylabel(sub_titles[i])
            
            if i == 0:
                plt.legend()
            
            
    #         plt.tight_layout()
    #         plt.savefig(os.path.join('Graph',f'{c}_{f}.png'))
    for ax in fig.get_axes():
    #     ax.label_outer()
        ax.tick_params(axis='x',labelrotation=90)
    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.2)
    for ax in axs.flatten():    
        ax.legend(loc=4,fontsize=8)
        ax.grid(False)
    plt.figtext(0.5,0.5,"(a)", va="center", ha="center", size=15)
    plt.figtext(0.5,0.01,"(b)", va="center", ha="center", size=15)
    fig.tight_layout(pad=3.0)
    # for i in range(4):
    #     axs[0,i].legend(loc=0,fontsize=8)
    # axs[0,2].legend(loc=4,fontsize=8)
    # axs[1,3].legend(loc=4)
    plt.savefig('F1.png',dpi=350)
analyse()

            
