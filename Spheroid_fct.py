# -------------------------------
# Spheroid_fct.py
# Author: Felix Romer
# Email: felix.lucas.romer@gmail.com
# Web: https://github.com/DaRoemer/Spheroid
# Date: 2022/08/15
# Version: 1.1
# Last Change: 2022/08/18
# -------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def multi_experiment_well_finder(file_list,Experiment_name, conditions, Correction_factor=1000,drop_args = [], printing=False):
    """Function to find the most representativ Spheroid over a specific time peroide
    -----------
        Args:
            file_list(list of str): List with Path of inpout files. File-type must be .xlsx
                            Use Provided Exel-file template from https://github.com/DaRoemer/Spheroid
                            See 'Read Me' for more informations
            Experient_name (list of str): Name of the Experiment to the coressenponding Excel file
            conditions (list of str): Conditions. Same order as in Excel file
            Correction_factor(int, opt., default=1000): Factor can be used to convert the Input-Values
            drop_args(list of str, opt.): Arguments from the 'commend'-Column of the exel file, which leed to a excusion of
                            the correspondention well at this spezific day 
    ------------
        Returns:
            mean_round_extended(DataFrame): DataFrame with the mean roundness of each Condition and day. On the values of the 1. experiment
                            follow the values of the 2. and so on.
            mean_area_extended (DataFrame): DataFrame with the mean area of each Condition and day. On the values of the 1. experiment
                            follow the values of the 2. and so on.
            rep_well_extendet (DataFrame): The representeativ well for each Condition and Experiment is shown in this DataFrame.
            df_adapt_extendet(DataFrame): Adapeted InputData. Includes column for corrected Lenth and Witch, Area, Roundness, Mean,... 
    """
    mean_round_extended=pd.DataFrame()
    mean_area_extended=pd.DataFrame()
    rep_well_extended=pd.DataFrame()
    df_adapt_extendet=pd.DataFrame()
    for file,Experiment in zip(file_list,Experiment_name):
        if printing == True:
            print(f'This are the results for {Experiment} from {file}')
        rep_wells,df_well, df_drop, df_adapt, end_wells, merged_df \
        =rep_well_finder(file, Experiment, conditions, Correction_factor, drop_args, printing)
        if printing == True:
            print('-------')
        mean_round_extended=pd.concat([mean_round_extended,merged_df])
        mean_area_extended=pd.concat([mean_area_extended,merged_df])
        rep_well_extended=pd.concat([rep_well_extended,rep_wells])
        df_adapt_extendet=pd.concat([df_adapt_extendet,df_adapt])
    return mean_round_extended,mean_area_extended,rep_well_extended, df_adapt_extendet

def plotfunc(mean_extended, Cell, Stat, color_palette=['#000000','#b3ffb3','#006600','#0000e6','#3366ff', '#ff0000', '#ffa500','#990000'],sharey=True,sharex=True):
    """Tool for easy plotting after using the 'multi_experiment_well_finder' function
    -----------
        Args:
            mean_extended (DataFrame): Mean-Statistic of interest. This is one of the DataFrames that was generated before
            Cell (str): Cellline that was used in the Experiment
            Stat (str): Value, that is used. 'Roundness' or 'Area'
            color_palette (list of str, optional): Color Palette is set for 8 Conditions! 
            sharey (Bool, opt): Determins whether y-axis is shared ('True') or whether every subplot get tis own y-axis ('False')
            sharex (Boll, opt): Determins whether x-axis is shared ('True') or whether every subplot get tis own x-axis ('False')
    ------------
        Returns:
            fig (matplotlib.pyplot.figure object): The final Plot
    """
    mean_extended.index.name = 'Day'
    mean_mean=mean_extended.groupby('Day').mean()

    mean_mean['Experiment']='Mean'
    mean_extended=pd.concat([mean_extended, mean_mean])
    mean_extended.reset_index(inplace=True)
    melt_round=mean_extended.melt(id_vars=['Day', 'Experiment'],var_name='Condition', value_name=Stat)

    sns.set_context('talk')
    fig=plt.figure(figsize=(10,10))
    

    fig=sns.relplot(data=melt_round,
                    x='Day',
                    y=Stat,
                    hue='Condition',
                    palette=color_palette,
                    col='Experiment',
                    kind='line',
                    facet_kws={'sharey':sharey, 'sharex':sharex}
    )
    fig.fig.suptitle(f'Mean {Stat} of {Cell}', y=1.07,x=0.44, size = 20)
    fig.set_titles('{col_name}',y=1.05)

    sns.move_legend(fig, "lower center", bbox_to_anchor=(0.44, -0.25),ncol=4, frameon=False)
    return fig


def Excel_file_sheet_reader(file):
    ''' Read Excel file and concate all sheets in one DataFrame
    ------
        Args:
            file (Excel.xlsx): Excel File with n sheet following the same structure
            sort_col (str, opt): Name of Column used to sort Data
            sheet_col (str, opt): Name of the additional Column holding the sheet names which will be added.
    ------
        Returns:
            df (df): Concated Data from Excel file with additional colum holdung the sheet names    
    '''
    xlsx_file = pd.ExcelFile(file)
    sheets = xlsx_file.sheet_names
    df=pd.DataFrame()
    for i, j in zip(sheets, np.arange(0,len(sheets))):
        df_sheet = pd.read_excel(file, sheet_name=i)
        df_sheet['day']=j
        df=pd.concat([df,df_sheet],axis=0)
    df.sort_values(['day','condition'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df=df[[c for c in df if c != 'day']+['day']]
    return df

# the following functions are optimised for the spheroid function and should not be used independently

def DataFrame_adapter(df, Korrekturfaktor, Experiment_name):
    df['Lenght_cor']=df['Length']/Korrekturfaktor
    df['Width_cor']=df['Width']/Korrekturfaktor
    df['Area_calc']=(df['Länge_ÜS']/2)*(df['Breite_ÜS']/2)*np.pi
    df['Vol_calc']=0.5*df.loc[:,'Länge_ÜS']*(df.loc[:,'Breite_ÜS']**2)
    df['Roundness']=df['Breite_ÜS']/df['Länge_ÜS']
    df['Experiment']=Experiment_name
    return df

def mean_calculater(df, df_drop, Col, Experiment_name, conditions):
    mean_list=[]
    for i in df['day'].unique():
        day_list=[]
        for cond in conditions:
            search_value=df_drop[(df_drop['day']==i)&(df_drop['condition']==cond)][Col].mean()
            day_list.append(search_value)
        mean_list.append(day_list)
    mean_df=pd.DataFrame(mean_list, columns=conditions)
    mean_df['Experiment']=Experiment_name
    return mean_df

def diff_to_mean(df, mean_df, Col,conditions):
    #find a rep well per Condition on the Volumen 
    # add column with differnece to the mean to df
    mean_liste=[]
    for day in df['day'].unique():
        for bed in conditions:
            for n in np.arange(0, (96/len(conditions))):
                mean_liste.append(mean_df.loc[day,bed])
    df[f"mean_{Col}"]=mean_liste
    df[f"diff_{Col}"]=(df[Col]-df[f"mean_{Col}"]).abs()
    return df
    
def rep(df_adapt):
    df_adapt['repressentation']=df_adapt['diff_Area']*df_adapt['diff_Roundness']
    return df_adapt

def well(df_rep, drop_args):    
    # df_well holds the mean diff to the mean Vol per well 
    df_well=pd.DataFrame((df_rep[~(df_rep.commend.isin(drop_args))]).groupby('Well')['repressentation'].mean())  
    return df_well

def drop_rows(df_well, drop_args):
    df_drop=df_well.loc[~(df_well.commend.isin(drop_args))]
    droped_well=df_well.loc[df_well.commend.isin(drop_args)]
    end_wells=list(droped_well[droped_well['day']==(len(df_drop['day'].unique())-1)]['Well'])
    return df_drop, end_wells

def print_result(df_well, end_wells,Experiment_name,conditions, printing):
#final result is printet per condition
    k=int(96/len(conditions))
    list_1=[['Condition', 'most representativ Well', 'repressentation factor']]
    for n in np.arange(0,len(conditions)):
        list_2=[]
        df_temp=df_well.loc[n*k+1:(n+1)*k]
        df_temp=df_temp[~(df_temp.index.isin(end_wells))]
        df_temp.reset_index(inplace=True)
        arg_min=np.argmin(df_temp['repressentation'])
        min_diff=df_temp.iloc[arg_min]['repressentation']
        well=int(df_temp.iloc[arg_min]['Well'])
        list_1.append([conditions[n], well, min_diff])
        if printing==True:
            print('The most representativ well for Condition', conditions[n], 'is well', well, '.')
            print('The mean difference to the mean Area times the mean difference to the mean roundness is', min_diff)
            print('---')
    rep_wells=pd.DataFrame(list_1[1:],columns=list_1[0])
    rep_wells['Experiment']=Experiment_name
    return rep_wells

def rep_well_finder(file,Experiment_name ,conditions,Correction_factor,drop_args,printing):
    df=Excel_file_sheet_reader(file)
    df_adapt=DataFrame_adapter(df, Correction_factor,Experiment_name)
    df_drop, end_wells=drop_rows(df_adapt, drop_args)

    condition_cols=['day', 'condition']
    mean_df = df_drop.groupby(condition_cols).mean().reset_index()
    value_cols = mean_df.columns
    merged_df = pd.merge(df, mean_df, on=condition_cols, suffixes=("", "_mean"))
    for col in value_cols:
        merged_df[f"{col}_diff"] = merged_df[col] - merged_df[f"{col}_mean"]
    
    merged_df['repressentation'] = merged_df['Area_calc_diff'] * merged_df['Roundness_diff']
    df_well=well(merged_df, drop_args)
    rep_wells=print_result(df_well, end_wells, Experiment_name,conditions, printing)
    return rep_wells,df_well, df_drop, df_adapt, end_wells, merged_df




        
  


