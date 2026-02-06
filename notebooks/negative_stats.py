from pathlib import Path
import re
from turtle import left
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from PIL import Image
from utils import *


def cleaned_data_negative():
    p = Path(__file__).resolve()
    base = None
    for parent in p.parents:
        if (parent / "data").exists():
            base = parent
            break
    if base is None:
        base = p.parents[3]
    raw = base / "data" / "raw"

    try:
      
        df_passing = pd.read_csv(raw / "passing_data.csv")
        df_adv_passing = pd.read_csv(raw / "adv_passing.csv")
        qb_stats_sumer = pd.read_csv(raw / "qb_stats_sumer.csv")

    except FileNotFoundError as e:
        print("Data file not found:", e)
        return
    
    df_passing = df_passing[df_passing['Pos'] == 'QB']
    df_passing = df_passing[df_passing['Team'] != '2TM']
    team_map = {
        'NWE': 'NE',
        'KAN': 'KC',
        'SFO': 'SF',
        'TAM': 'TB',
        'GNB': 'GB',
        'NOR': 'NO',
        'LVR': 'LV',

    }
    df_passing['Team'] = df_passing['Team'].replace(team_map)
    df_passing['Att'] = df_passing['Att'].str.replace(r'*', '', regex=False)
    df_passing['Att'] = pd.to_numeric(df_passing['Att'], errors='coerce').fillna(0).astype(int)

    qb_stats_sumer = qb_stats_sumer.rename(columns = {'Player Name': 'Player'})
    qb_stats_sumer['Player'] = qb_stats_sumer['Player'].str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
    qb_stats_sumer = qb_stats_sumer.rename({'Cameron Ward': 'Cam Ward'})

    df_adv_passing= promote_first_row_to_header(df_adv_passing)
    df_adv_passing = df_adv_passing[df_adv_passing['Team'] != '2TM']
    print(df_adv_passing[df_adv_passing['Player'] == 'J.J. McCarthy']['Bad%'])
    df_final_negative = df_passing[['Player', 'Team', 'Att']].merge(df_adv_passing[['Player', 'Bad%']], on='Player', how='left')
    df_final_negative = qb_stats_sumer[['Player', 'Time To Throw']].merge(df_final_negative, on='Player', how='left')
    df_final_negative = df_final_negative[df_final_negative['Att'] >= 200]
    df_final_negative['Time To Throw'] = pd.to_numeric(df_final_negative['Time To Throw'], errors='coerce').astype(float)
    df_final_negative['Bad%'] = pd.to_numeric(df_final_negative['Bad%'], errors='coerce').astype(float)
    df_final_negative = df_final_negative.drop_duplicates(subset=['Player'])
    print(df_final_negative.head())
    return df_final_negative

def create_scatter_negative(df):
    x = df['Time To Throw']
    y = df['Bad%']
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    df['dist'] = np.sqrt((x - x_mean)**2 + (y - y_mean)**2)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(x, y, alpha=0, s=30, c='gray', zorder=1)
    
    logo_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "nfl logos"
    
    for idx, row in df.iterrows():
        current_x = row['Time To Throw']
        current_y = row['Bad%']
        
        logo_file = logo_path / f"{row['Team']}.png"
        
        z_val = 10 + int(row['dist'] * 100)
        
        if logo_file.exists():
            img = Image.open(logo_file).convert("RGBA")
            imagebox = OffsetImage(img, zoom=0.009) 
            ab = AnnotationBbox(imagebox, (current_x, current_y), 
                               frameon=False, zorder=z_val)
            ax.add_artist(ab)
            
        if row['dist'] > 0.04: 
            ax.annotate(row['Player'], (current_x, current_y), 
                       fontsize=6.5, fontfamily='Times New Roman', fontweight='semibold', alpha=0.9,
                       xytext=(0, 10), textcoords='offset points', 
                       ha='center', zorder=z_val + 1)

    ax.set_xlabel('Time To Throw (Seconds)', fontsize = 16, fontfamily='Times New Roman')
    ax.set_ylabel('Bad% (Bad Throw Percentage)', fontsize=16, fontfamily='Times New Roman')
    ax.set_title('NFL QBs: Standard (Min 200 Att)', fontsize=20, fontweight = 'semibold', alpha = 0.7, pad=20, fontfamily='Times New Roman')

    ax.axhline(y=y_mean, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=x_mean, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    ax.text(
    x=0+0.88,              
    y=y_mean+0.15,            
    s=f'Avg BadTh%: {y_mean:.2f}', 
    color='black',
    va='center',         
    ha='left',           
    fontsize=9,
    alpha=0.9,
    transform=ax.get_yaxis_transform() 
)

    ax.text(
    x=x_mean+0.051,            
    y=0+0.003,             
    s=f'Avg Time to Throw: {x_mean:.2f}', 
    color='black',
    va='bottom',         
    ha='center',         
    fontsize=9,
    alpha=0.9,
    transform=ax.get_xaxis_transform()
)
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()

    bad_rect = patches.Rectangle((x_mean, y_mean), right-x_mean, top - y_mean, 
                               linewidth=0, facecolor='red', alpha=0.08, zorder=0)
    ax.add_patch(bad_rect)

    good_rect = patches.Rectangle((left, bottom),   x_mean - left, y_mean - bottom, 
                               linewidth=0, facecolor='green', alpha=0.08, zorder=0)
    ax.add_patch(good_rect)

    plt.tight_layout()
    plt.savefig("caleb_negative_stats.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df = cleaned_data_negative()
    create_scatter_negative(df)